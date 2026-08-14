from rag.embedder import Embedder
from rag.vectorstore import VectorStore

SIMILARITY_THRESHOLD = 0.35


class Retriever:
    def __init__(self , embedder:Embedder , vectorstore:VectorStore):
        self.embedder = embedder
        self.vector_store = vectorstore

    def retrieve(self , query:str,top_k:int=5)-> list[dict]:
        """Embed the query and retrieve top_k chunks above similarity threshold"""

        query_vec = self.embedder.embed_query(query)
        results = self.vector_store.query(query_vec,top_k=top_k)


        if not results["documents"] or not results["documents"][0]:
            print("No results found.")
            return []

        docs = [
        {

            "id":doc_id,
            "text":text,
            "metadata":metadata,
            "score":round(1-distance,4)
        }

        for doc_id , text , metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
        if (1 - distance) >= SIMILARITY_THRESHOLD
       ]

        print(f"Retrieval {len(docs)} relevant chunks for query")
        return docs    
