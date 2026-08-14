import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    MODEL_NAME = "BAAI/bge-small-en-v1.5"
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


    def __init__(self):
        print(f"Loading embedding model:{self.MODEL_NAME}")
        self.model = SentenceTransformer(self.MODEL_NAME)
        print(f"Embedding dimensions : {self.model.get_embedding_dimension()}")

    def embed_batch(self,texts:list[str]) -> np.ndarray:
        """EMbed a list of texts (used during ingestion)"""
        return np.array(self.model.encode(texts ,show_progress_bar=True,batch_size=32))  #type:ignore[arg-type]

    def embed_query(self,query:str) -> list[float]:
        """Embed a single query string (used during retrieval)"""

        return list(map(float,self.model.encode([self.QUERY_PREFIX + query])[0])) #type:ignore[arg-type]