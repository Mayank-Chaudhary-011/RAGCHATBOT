from rag.loader import load_docx_folder
from rag.chunker import chunk_documents
from rag.embedder import Embedder
from rag.vectorstore import VectorStore

def main():
    print("=" * 40)
    print(" 3GPP RAG Ingestion Pipeline")
    print("=" * 40)

    print("\n[1/5] Loading .docx document...")
    documents = load_docx_folder("docs")

    print("\n[2/5] Chunking documents...")
    chunks = chunk_documents(documents)

    print("\n[3/5] Loading embedding model...")
    embedder = Embedder()

    print("\n[4/5] Generating embeddings...")
    embeddings = embedder.embed_batch([chunk.page_content for chunk in chunks])


    print("\n[5/5] Storing in ChromaDB...")
    store = VectorStore()
    if store.count() > 0:
        print(f"Vector store already has {store.count()} docs. Skipping.")
        return

    store.add_documents(chunks, embeddings)
    print("\n Ingestion completed! Ready to chat.")


if __name__ == "__main__":
    main()