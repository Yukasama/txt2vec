# Training-Modul

Dieses Modul enthält die gesamte Logik für das Training von SBERT-Modellen (Sentence-BERT) inklusive Datenvalidierung, Trainingsorchestrierung, Fehlerbehandlung und Utilities für die Integration in REST-APIs.

## Übersicht

Das Trainingssystem ist modular aufgebaut und bietet:
- **Datenvalidierung**: Sicherstellung, dass Trainingsdaten korrekt und vollständig sind
- **Trainingsorchestrierung**: Kapselung des gesamten Trainingsablaufs (Vorbereitung, Training, Speichern, Aufräumen)
- **Fehlerbehandlung**: Klare Fehlerklassen für alle typischen Fehlerfälle
- **Integration**: Einfache Anbindung an FastAPI-Endpunkte und Datenbank

## Architektur

### Hauptkomponenten

1. **`training_orchestrator.py`**: Zentrale Trainingslogik
   - Klasse `TrainingOrchestrator` kapselt den gesamten Ablauf
   - Methoden für Datenvorbereitung, Training, Speichern, Fehlerbehandlung und Aufräumen

2. **`service.py`**: Service-Layer für API-Integration
   - Validierung von Requests
   - Aufruf des Orchestrators
   - Fehler- und Statusmanagement

3. **`utils/`**: Hilfsfunktionen
   - **`validators.py`**: Validierung von Trainingsdaten und Parametern
   - **`input_examples.py`**: Erstellung und Filterung von Trainingsbeispielen
   - **`model_loader.py`**: Laden und Initialisieren von Modellen
   - **`cleanup.py`**: Ressourcenmanagement und Aufräumarbeiten
   - **`uuid_validator.py`**: UUID-Validierung für Dataset- und Model-IDs
   - **`safetensors_finder.py`**: Auffinden von Modell-Dateien

## Beispiel: Training per Python

```python
from vectorize.training.training_orchestrator import TrainingOrchestrator
from vectorize.training.schemas import TrainRequest
from sqlmodel.ext.asyncio.session import AsyncSession
from uuid import uuid4

# Training-Request erstellen
train_request = TrainRequest(
    model_tag="sentence-transformers/all-MiniLM-L6-v2",
    train_dataset_ids=["uuid-des-datensatzes"],
    epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
)

# Training-Orchestrator initialisieren
db = AsyncSession()  # Ihre DB-Session
task_id = uuid4()
orchestrator = TrainingOrchestrator(db, task_id)

# Training starten (asynchron)
await orchestrator.run_training(
    model_path="data/models/base-model",
    train_request=train_request,
    dataset_paths=["data/datasets/train.jsonl"],
    output_dir="data/models/my-trained-model"
)
```

## Beispiel: Training per REST-API

Das Training wird als Background-Task gestartet und liefert eine Task-ID zurück:

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model_tag": "sentence-transformers/all-MiniLM-L6-v2",
    "train_dataset_ids": ["uuid-des-datensatzes"],
    "epochs": 3,
    "per_device_train_batch_size": 16,
    "learning_rate": 2e-5,
    "warmup_steps": 100,
    "evaluation_steps": 500
  }'
```

Response:
```json
{
  "task_id": "uuid-der-training-task",
  "status": "running",
  "message": "Training gestartet"
}
```

### Status abfragen

```bash
curl "http://localhost:8000/train/status/{task_id}"
```

### Alle verfügbaren Parameter

Die API unterstützt alle wichtigen Sentence-Transformers Parameter:
- `model_tag`: Basis-Modell aus der Datenbank
- `train_dataset_ids`: Liste von Dataset-IDs (werden zusammengeführt)
- `val_dataset_id`: Optional, für Validierung
- `epochs`, `per_device_train_batch_size`, `learning_rate`
- `warmup_steps`, `optimizer_name`, `scheduler`
- `weight_decay`, `max_grad_norm`, `use_amp`
- `evaluation_steps`, `save_best_model`, `timeout_seconds`

## Fehlerbehandlung

Das System bietet detaillierte Fehlerklassen für verschiedene Szenarien:

- **`InvalidDatasetIdError`**: Ungültige UUID oder fehlende Dataset-ID
- **`InvalidModelIdError`**: Ungültiges Modell oder Modell-Tag nicht gefunden
- **`TrainingDatasetNotFoundError`**: Trainingsdatensatz existiert nicht im Dateisystem
- **`TrainingModelWeightsNotFoundError`**: Modellgewichte nicht gefunden
- **`TrainingTaskNotFoundError`**: Training-Task mit gegebener ID nicht gefunden
- **`DatasetValidationError`**: Datensatz hat falsche Struktur oder fehlerhafte Daten
- **Timeout-Errors**: Training überschreitet das definierte Zeitlimit

Alle Fehler werden detailliert geloggt und über die API zurückgegeben.

## Best Practices

### Datenqualität
- **Vor dem Training validieren**: Nutze die eingebauten Validatoren
- **Ausgewogene Daten**: Sorge für gute positive/negative Beispiele
- **Datenqualität prüfen**: Positive Beispiele sollten semantisch ähnlich sein

### Modell-Management
- **Eindeutige Model-Tags verwenden**: Verhindert Überschreibungen
- **Basis-Modelle aus HuggingFace Hub**: z.B. `sentence-transformers/all-MiniLM-L6-v2`
- **Validierungsdaten bereitstellen**: Für bessere Trainingsüberwachung

### Performance & Ressourcen
- **Batch-Größe anpassen**: Je nach verfügbarem GPU-Speicher
- **Warmup-Steps nutzen**: Verbessert Trainingsstabilität
- **Timeout setzen**: Verhindert hängende Training-Jobs
- **Ressourcen nach Training freigeben**: Automatisches CUDA-Cleanup

## Was sind Sentence-Transformers?

[Sentence-Transformers](https://www.sbert.net/) sind spezialisierte Modelle, die auf BERT, RoBERTa oder ähnlichen Architekturen basieren und darauf trainiert sind, ganze Sätze oder Textabschnitte als dichte Vektoren (Embeddings) im semantischen Raum abzubilden. Dadurch können semantisch ähnliche Sätze durch ähnliche Vektoren repräsentiert werden.

Typische Anwendungsfälle:
- Semantische Suche (Semantic Search)
- Clustering und Klassifikation von Texten
- Ähnlichkeitsmessung (z.B. Duplicate Detection)

## Trainingsprinzip & Loss-Funktion

Das Training erfolgt meist mit sogenannten **Triplet- oder Pairwise-Losses**. Im Standardfall werden Triplets bestehend aus:
- **Anchor (Question)**: Ausgangsfrage oder Satz
- **Positive**: Semantisch ähnliche Antwort
- **Negative**: Semantisch unähnliche Antwort
verwendet.

### Typische Loss-Funktionen

- **CosineSimilarityLoss**: Maximiert die Kosinus-Ähnlichkeit zwischen Anchor und Positive und minimiert sie zwischen Anchor und Negative.
- **MultipleNegativesRankingLoss**: Effizienter Pairwise-Loss, der für große Batches geeignet ist.
- **TripletLoss**: Klassischer Triplet-Loss, der einen Margin zwischen positiven und negativen Paaren erzwingt.

Beispiel (CosineSimilarityLoss):
```python
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

model = SentenceTransformer('distilbert-base-nli-mean-tokens')
train_examples = [
    InputExample(texts=["Frage", "Positive Antwort", "Negative Antwort"]),
    # ...
]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100
)
```

**Ziel:**
- Die Embeddings von Anchor und Positive sollen möglichst ähnlich (hohe Kosinus-Ähnlichkeit) sein.
- Die Embeddings von Anchor und Negative möglichst unähnlich (niedrige Kosinus-Ähnlichkeit).

Weitere Details und Loss-Varianten siehe [SBERT-Dokumentation](https://www.sbert.net/docs/package_reference/losses.html).

## Testen

Tests für das Trainingsmodul befinden sich in `tests/training/`:

```bash
pytest tests/training/ -v
```

- **Valid-Tests**: Erfolgreiches Training mit korrekten Daten
- **Invalid-Tests**: Fehlerfälle und fehlerhafte Daten

## Abhängigkeiten

- **sentence-transformers**: Kernbibliothek für SBERT-Training
- **torch**: PyTorch Backend für das Training
- **pandas**: Datenhandling und -validierung
- **numpy**: Numerische Operationen
- **loguru**: Strukturiertes Logging
- **sqlmodel**: Datenbankoperationen
- **fastapi**: REST-API Framework
- **pydantic**: Datenvalidierung und Serialisierung

## Datenformat

Das System erwartet JSONL-Dateien mit folgender Struktur:

```json
{"question": "Was ist maschinelles Lernen?", "positive": "ML ist ein Teilbereich der KI", "negative": "Das Wetter ist heute schön"}
{"question": "Wie funktioniert Deep Learning?", "positive": "Mit neuronalen Netzen", "negative": "Pizza schmeckt gut"}
```

**Erforderliche Spalten:**
- `question`: Die Ausgangsfrage oder der Anchor
- `positive`: Semantisch ähnliche/korrekte Antwort
- `negative`: Semantisch unähnliche/falsche Antwort

**Validierung:**
- Alle Spalten müssen vorhanden sein
- Keine NULL-Werte oder leere Strings
- Mindestens ein Trainingseintrag erforderlich

---

Für Details siehe auch die Docstrings in den jeweiligen Modulen und die API-Dokumentation (Swagger/OpenAPI).
