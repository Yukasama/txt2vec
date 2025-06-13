"""Service for importing GitHub models."""

import tempfile
from pathlib import Path

import httpx
from fastapi import status
from git import Repo
from loguru import logger

from .exceptions import InvalidModelError, ModelNotFoundError, NoValidModelsFoundError

__all__ = ["load_github_model_and_cache_only_svc", "repo_info"]

_github_models: dict[str, str] = {}


def load_github_model_and_cache_only_svc(
    owner: str,
    repo: str,
    branch: str = "main",
) -> dict[str, str]:
    """Lädt ein GitHub Repo und cacht das Modell.

    Klont ein GitHub-Repo sucht darin nach genau einer
    „pytorch_model.bin“, „config.json“
    und „tokenizer.json“ und cached sie auf der Festplatte sowie im Memory.

    Args:
        owner: GitHub-Username oder Organisation (z. B. "domoar")
        repo:  Repo-Name (z. B. "BogusModel")
        branch: Branch- oder Tag-Name (default: "main")

    Returns:
        {
            "owner": owner,
            "repo": repo,
            "branch": branch,
            "local_path": "<Pfad zur pytorch_model.bin-Datei>",
            "key": "owner/repo@branch"
        }

    Raises:
        NoValidModelsFoundError:   Wenn nicht genau je eine Datei gefunden wurde.
        InvalidModelError:         Für alle anderen Fehler (Clone-Fehler, IO, …).
    """
    key = f"{owner}/{repo}@{branch}"

    if key in _github_models:
        logger.info("GitHub-Model bereits im In-Memory-Cache.", modelKey=key)
        return {
            "owner": owner,
            "repo": repo,
            "branch": branch,
            "local_path": _github_models[key],
            "key": key,
        }

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_url = f"https://github.com/{owner}/{repo}.git"
            Repo.clone_from(repo_url, tmpdir, branch=branch)
            repo_obj = Repo(tmpdir)
            repo_obj.git.checkout(branch)

            base = Path(tmpdir)

            checks: list[tuple[str, str]] = [
                ("pytorch_model.bin", "Model-Datei"),
                ("config.json", "Config-Datei"),
                ("tokenizer.json", "Tokenizer-Datei"),
            ]

            paths: dict[str, Path] = {}
            for filename, description in checks:
                matches = list(base.rglob(filename))
                if len(matches) != 1:
                    raise NoValidModelsFoundError(
                        f"{len(matches)} {description}(en) gefunden für {filename}"
                    )
                paths[filename] = matches[0]
                logger.debug("Gefundene {}: {}", description, matches[0])

            cache_dir = Path("gh_cache") / f"{owner}_{repo}@{branch}"
            cache_dir_parent = cache_dir.parent
            cache_dir_parent.mkdir(parents=True, exist_ok=True)

            if cache_dir.exists():
                logger.info("Model bereits lokal gecached unter {}", cache_dir)
            else:
                logger.info("Cache-Verzeichnis anlegen: {}", cache_dir)
                cache_dir.mkdir()

                for filename, pfad in paths.items():
                    dest = cache_dir / filename
                    pfad.replace(dest)
                    paths[filename] = dest

            lokal_pfad = str((cache_dir / "pytorch_model.bin").resolve())
            _github_models[key] = lokal_pfad

        logger.info("GitHub-Model erfolgreich gecached.", modelKey=key)
        return {
            "owner": owner,
            "repo": repo,
            "branch": branch,
            "local_path": lokal_pfad,
            "key": key,
        }

    except NoValidModelsFoundError:
        raise
    except Exception as e:
        logger.exception("Fehler beim Laden des GitHub-Models.", modelKey=key)
        raise InvalidModelError(f"GitHub-Import fehlgeschlagen: {e}") from e


def repo_info(repo_url: str, revision: str | None = None) -> bool:
    """Check if ein GitHub-Repo und Branch/Tag existiert.

    Verwendet GitHub-API: /repos/{owner}/{repo}/branches/{branch}

    Args:
        repo_url (str): HTTPS-URL des GitHub-Repos (z.B. "https://github.com/user/repo").
        revision (str | None): Branch- oder Tag-Name. Falls None, default "main".

    Returns:
        bool: True, wenn Repo + Branch/Tag existieren.

    Raises:
        ModelNotFoundError: Wenn Repo oder Branch/Tag nicht gefunden werden.
    """
    api_url = str(repo_url).replace(
        "https://github.com/", "https://api.github.com/repos/"
    ).rstrip("/")
    branch = revision or "main"
    check_url = f"{api_url}/branches/{branch}"
    resp = httpx.get(check_url, timeout=10)
    if resp.status_code != status.HTTP_200_OK:
        raise ModelNotFoundError(check_url)
    return True
