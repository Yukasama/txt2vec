"""Einfacher Test für Embedding Performance
Lädt HuggingFace-Modell und misst Inference-Geschwindigkeit
"""

import asyncio
import time

import httpx


class EmbeddingPerformanceTest:
    def __init__(self, base_url="http://localhost:8000/v1"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=300.0)  # 5 Minuten Timeout

    async def close(self):
        await self.client.aclose()

    async def load_huggingface_model(self, model_tag, revision="main"):
        """Lädt ein HuggingFace-Modell über den API-Endpoint"""
        print(f"Lade Modell: {model_tag}")

        upload_data = {"model_tag": model_tag, "revision": revision}

        start_time = time.time()

        try:
            response = await self.client.post(
                f"{self.base_url}/uploads/huggingface", json=upload_data
            )

            if response.status_code == 201:
                load_time = time.time() - start_time
                print(f"Upload-Request erfolgreich in {load_time:.2f}s")

                # Task ID extrahieren
                location = response.headers.get("Location", "")
                task_id = location.split("/")[-1] if location else None

                return await self._wait_for_model_ready(model_tag, revision)

            if response.status_code == 409:
                print(f"Modell bereits vorhanden: {model_tag}")
                model_key = f"{model_tag}@{revision}"

                # Auch bei vorhandenem Modell testen ob es bereit ist
                print("Teste ob vorhandenes Modell bereit ist...")
                if await self._test_model_ready(model_key):
                    print(f"Modell ist bereit: {model_key}")
                    return model_key
                print("Modell lädt noch, warte...")
                return await self._wait_for_model_ready(model_tag, revision)

            print(f"Upload fehlgeschlagen: {response.status_code}")
            print(f"Response: {response.text}")
            return None

        except Exception as e:
            print(f"Fehler beim Upload: {e}")
            return None

    async def _wait_for_model_ready(self, model_tag, revision, max_wait=300):
        """Wartet bis das Modell in der Modell-Liste verfügbar ist"""
        model_key = f"{model_tag}@{revision}"
        print(f"Warte auf Modell-Loading: {model_key}")

        start_wait = time.time()

        while time.time() - start_wait < max_wait:
            try:
                # Prüfe verfügbare Modelle
                models_response = await self.client.get(
                    f"{self.base_url}/models?size=50"
                )

                if models_response.status_code == 200:
                    models_data = models_response.json()

                    # Suche nach unserem Modell
                    for model in models_data.get("items", []):
                        if model.get("model_tag") == model_key:
                            wait_time = time.time() - start_wait
                            print(f"Modell in Liste nach {wait_time:.2f}s: {model_key}")

                            # Warmup-Request zum Testen der Bereitschaft
                            print("Teste Model-Bereitschaft mit Warmup-Request...")
                            if await self._test_model_ready(model_key):
                                print(f"Modell ist bereit: {model_key}")
                                return model_key
                            print("Modell noch nicht bereit, warte weitere 10s...")
                            await asyncio.sleep(10)

                # Alle 5 Sekunden prüfen
                await asyncio.sleep(5)
                print(f"Warte noch... ({time.time() - start_wait:.0f}s)")

            except Exception as e:
                print(f"Fehler beim Prüfen: {e}")
                await asyncio.sleep(5)

        print(f"Timeout nach {max_wait}s - Modell eventuell noch nicht bereit")
        return model_key  # Versuchen trotzdem

    async def _test_model_ready(self, model_key):
        """Testet ob das Modell wirklich bereit ist mit einem einfachen Request"""
        try:
            warmup_data = {"model": model_key, "input": "test"}

            response = await self.client.post(
                f"{self.base_url}/embeddings", json=warmup_data
            )

            return response.status_code == 200

        except Exception:
            return False

    async def test_embedding_speed(self, model_tag, test_texts, iterations=5):
        """Testet die Geschwindigkeit der Embedding-Generierung"""
        print(f"\nTeste Embedding-Geschwindigkeit: {model_tag}")
        print(f"{len(test_texts)} Texte, {iterations} Durchläufe")

        times = []

        for i in range(iterations):
            print(f"  Durchlauf {i + 1}/{iterations}...", end=" ")

            embedding_data = {"model": model_tag, "input": test_texts}

            start_time = time.time()

            try:
                response = await self.client.post(
                    f"{self.base_url}/embeddings", json=embedding_data
                )

                end_time = time.time()
                response_time = end_time - start_time

                if response.status_code == 200:
                    times.append(response_time)
                    print(f"{response_time:.3f}s OK")
                else:
                    print(f"Fehler {response.status_code}: {response.text}")

            except Exception as e:
                print(f"Fehler: {e}")

            # Kurze Pause zwischen Tests
            await asyncio.sleep(0.5)

        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)

            print(f"\nErgebnisse für {model_tag}:")
            print(f"   Durchschnitt: {avg_time:.3f}s")
            print(f"   Minimum:      {min_time:.3f}s")
            print(f"   Maximum:      {max_time:.3f}s")
            print(f"   Alle Zeiten:  {[f'{t:.3f}s' for t in times]}")

            return {
                "model": model_tag,
                "average": avg_time,
                "min": min_time,
                "max": max_time,
                "times": times,
            }

        return None


async def main():
    """Hauptfunktion für den Performance-Test"""
    # Test-Konfiguration
    model_to_test = "sentence-transformers/all-MiniLM-L6-v2"

    # Test-Texte (verschiedene Größen)
    test_cases = {
        "Einzeltext": ["This is a simple test sentence for embedding generation."],
        "Batch_5": [
            "This is the first test sentence.",
            "Here comes the second example text.",
            "A third sentence for batch processing.",
            "Fourth text in this batch request.",
            "Finally, the fifth and last sentence.",
        ],
        "Langer_Text": [
            "This is a much longer text that contains multiple sentences and should test how the embedding generation performs with longer input sequences. The text includes various topics such as machine learning, natural language processing, deep learning architectures, transformer models, attention mechanisms, and their applications in modern AI systems."
        ],
    }

    tester = EmbeddingPerformanceTest()

    try:
        print("Embedding Performance Test")
        print("=" * 50)

        # 1. Modell laden
        model_tag = await tester.load_huggingface_model(model_to_test)

        if not model_tag:
            print("Modell konnte nicht geladen werden - Test abgebrochen")
            return

        print(f"\nModell bereit: {model_tag}")

        # 2. Performance-Tests durchführen
        all_results = {}

        for test_name, test_texts in test_cases.items():
            print(f"\n{'-' * 30}")
            result = await tester.test_embedding_speed(model_tag, test_texts)
            if result:
                all_results[test_name] = result

        # 3. Zusammenfassung
        print(f"\n{'=' * 50}")
        print("ZUSAMMENFASSUNG")
        print(f"{'=' * 50}")

        for test_name, result in all_results.items():
            print(
                f"{test_name:15}: {result['average']:.3f}s (+/-{result['max'] - result['min']:.3f}s)"
            )

        # 4. Für Vergleich mit alter Version
        print("\nFür Branch-Vergleich:")
        print("   1. Branch wechseln: git checkout <alter-branch>")
        print("   2. Server neu starten: uv run app")
        print("   3. Diesen Test erneut ausführen")
        print("   4. Ergebnisse vergleichen")

    finally:
        await tester.close()


if __name__ == "__main__":
    # Test ausführen
    print("Starting Embedding Performance Test...")
    asyncio.run(main())
