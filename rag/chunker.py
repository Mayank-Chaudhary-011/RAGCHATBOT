from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_documents(
        documents:list[Document],
        chunk_size: int=800,
        chunk_overlap:int=100

) -> list [Document]:

    """Split 3GPP Documents into overlapping chunks.
       Larger Chunk_size(800) preserves technical context better than 500
    """

    splitter =  RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators=["\n\n","\n",". "," ",""]
    )

    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks