# ruff: noqa: S101
"""Github endpoint invalid input checks."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

_REPO_OWNER = "domoar"
_REPO_NAME = "BogusModel"
_BRANCH_DEFAULT = "main"
_BRANCH_DIFF = "DifferentBranch-Full"


@pytest.mark.github
def test_load_bogus_model_tag_and_conflict_on_empty_tag(client: TestClient) -> None:
    """Tests that uploading the same model with and without tag causes conflict."""
    payload_main = {
        "owner": _REPO_OWNER,
        "repo_name": _REPO_NAME,
        "tag": _BRANCH_DEFAULT
    }

    response_main = client.post("uploads/github", json=payload_main)
    assert response_main.status_code == status.HTTP_201_CREATED

    payload_empty = {
        "owner": _REPO_OWNER,
        "repo_name": _REPO_NAME,
        "tag": ""
    }
    response_empty = client.post("uploads/github", json=payload_empty)
    assert response_empty.status_code == status.HTTP_409_CONFLICT


@pytest.mark.github
def test_load_bogus_model_branch_tag(client: TestClient) -> None:
    """Tests default main branch.

    Args:
        client (TestClient): _description_
    """
    payload = {
        "owner": _REPO_OWNER,
        "repo_name": _REPO_NAME,
        "tag": _BRANCH_DIFF
    }
    response = client.post("uploads/github", json=payload)
    assert response.status_code == status.HTTP_201_CREATED
