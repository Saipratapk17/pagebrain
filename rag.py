from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

# Browser-like headers so websites don't block the scraper
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

llm = None
vector_store = None


def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.9, max_tokens=500)

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR)
        )


def is_pdf(source):
    """Check if a source is a PDF — either a URL ending in .pdf or a local file path"""
    return source.lower().endswith(".pdf")


def load_source(source):
    """
    Load a single source — automatically detects if it's a PDF or webpage
    :param source: URL or local file path
    :return: list of Document objects
    """
    if is_pdf(source):
        # PDF — works for both online PDF URLs and local PDF files
        loader = PyPDFLoader(source)
    else:
        # Webpage — uses browser headers to avoid getting blocked
        loader = WebBaseLoader(
            web_paths=[source],
            requests_kwargs={"headers": HEADERS}
        )
    return loader.load()


def process_urls(urls):
    """
    This function scraps data from urls/pdfs and stores it in a vector db
    :param urls: input urls or file paths (can be mix of webpages and PDFs)
    :return:
    """
    yield "Initializing Components"
    initialize_components()

    yield "Resetting vector store...✅"
    vector_store.reset_collection()

    yield "Loading data...✅"
    all_data = []
    for source in urls:
        try:
            docs = load_source(source)
            all_data.extend(docs)
            yield f"Loaded: {source[:60]}...✅"
        except Exception as e:
            yield f"Failed to load: {source[:60]} — {str(e)}"

    if not all_data:
        yield "❌ No content extracted from any source. Try different URLs."
        return

    yield f"Total pages/documents loaded: {len(all_data)}...✅"

    yield "Splitting text into chunks...✅"
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE
    )
    docs = text_splitter.split_documents(all_data)

    yield f"Total chunks created: {len(docs)}...✅"

    yield "Add chunks to vector database...✅"
    if not docs:
        yield "❌ No content extracted. Try a different URL or PDF."
        return

    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)

    yield "Done adding docs to vector database...✅"


def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector database is not initialized ")

    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
    result = chain.invoke({"question": query}, return_only_outputs=True)
    sources = result.get("sources", "")

    return result['answer'], sources


if __name__ == "__main__":
    # Mix of webpage + PDF URL — both work now!
    urls = [
        "https://en.wikipedia.org/wiki/Amazon_Redshift",
        "https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf"
    ]

    for status in process_urls(urls):
        print(status)

    answer, sources = generate_answer("What is the attention mechanism?")
    print(f"Answer: {answer}")
    print(f"Sources: {sources}")
