import os 
import uuid
import chromadb
from langchain_core.documents import Document

COLLECTION_NAME = "3gpp_specs"
PERSIST_DIR = "chroma_db"


class VectorStore:
    def __init__(self):
        os.makedirs(PERSIST_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(path=PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space":"cosine",
                "description":"3GPP Telecom Standards RAG Collection"})
        print(f"VectorStore ready | Collection: {COLLECTION_NAME} | Docs : {self.collection.count()}")

    def add_documents(self,documents:list[Document],embeddings) -> None:
        """Store documents chunks with their embedding into chromaDB"""

        ids , texts , metadatas , vecs = [], [],[],[]

        for doc , emb in zip(documents , embeddings):
            ids.append(f"doc_{uuid.uuid4()}")
            texts.append(doc.page_content)
            metadatas.append({k:str(v) for k , v in doc.metadata.items()})
            vecs.append(emb)

        self.collection.add(ids=ids , documents=texts , metadatas=metadatas , embeddings=vecs)
        print(f"Added {len(ids)} Chunks | Total in store : {self.collection.count()}")

    def query(self , query_embedding :list[float],top_k:int = 5) -> dict:
        """Retrieve top_k most similar chunks from chromaDB"""

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents","metadatas","distances"]
        )

    def count(self) -> int:
        return self.collection.count()