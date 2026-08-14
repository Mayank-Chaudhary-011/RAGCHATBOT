import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"

class Embedder:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("NVIDIA_API_KEY"),
            base_url="https://integrate.api.nvidia.com/v1"
        )
        print(f"Embedder ready | Model: {EMBED_MODEL}")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        result = []
        batch_size = 10
        total = len(texts)

        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            print(f"  Embedding chunks {i+1}-{min(i+batch_size, total)} of {total}...")
            resp = self.client.embeddings.create(
                input=batch,
                model=EMBED_MODEL,
                encoding_format="float",
                extra_body={"input_type": "passage", "truncate": "END"}
            )
            for item in resp.data:
                result.append(item.embedding)

        return result

    def embed_query(self, query: str) -> list[float]:
        resp = self.client.embeddings.create(
            input=query,
            model=EMBED_MODEL,
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "END"}
        )
        return resp.data[0].embedding
