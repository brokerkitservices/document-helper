import asyncio
import os
import ssl
from typing import List

import certifi
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import Colors, log_error, log_header, log_info, log_success, log_warning

# Load environment variables
load_dotenv()

# Configuration Constants
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 200
BATCH_SIZE = 500
CRAWL_URL = "https://python.langchain.com/"
MAX_DEPTH = 2
EXTRACT_DEPTH = "advanced"
EMBEDDING_MODEL = "text-embedding-3-small"


# Initialization Functions
def setup_ssl():
    """Configure SSL context to use certifi certificates."""
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    return ssl_context


def get_embeddings():
    """Initialize and return OpenAI embeddings."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        show_progress_bar=False,
        chunk_size=50,
        retry_min_seconds=10,
    )


def get_vectorstore(embeddings=None):
    """Initialize and return Pinecone vector store."""
    if embeddings is None:
        embeddings = get_embeddings()

    return PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME", "dochelp"), embedding=embeddings
    )


def get_tavily_tools():
    """Initialize and return Tavily tools."""
    return {
        "extract": TavilyExtract(),
        "map": TavilyMap(max_depth=5, max_breadth=20, max_pages=1000),
        "crawl": TavilyCrawl(),
    }


# Core Functions
async def add_batch(
    vectorstore, batch: List[Document], batch_num: int, total_batches: int
):
    """Add a single batch of documents to the vector store."""
    try:
        await vectorstore.aadd_documents(batch)
        log_success(
            f"VectorStore Indexing: Successfully added batch {batch_num}/{total_batches} ({len(batch)} documents)"
        )
        return True
    except Exception as e:
        log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
        return False


async def index_documents_async(
    documents: List[Document], batch_size: int = BATCH_SIZE
):
    """Process documents in batches asynchronously."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    # Initialize vectorstore
    vectorstore = get_vectorstore()

    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    # Process batches concurrently
    tasks = [
        add_batch(vectorstore, batch, i + 1, len(batches))
        for i, batch in enumerate(batches)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
):
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(documents)} documents with {chunk_size} chunk size and {chunk_overlap} overlap",
        Colors.YELLOW,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    splitted_docs = text_splitter.split_documents(documents)

    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(documents)} documents"
    )

    return splitted_docs


def crawl_documentation(
    url: str = CRAWL_URL, max_depth: int = MAX_DEPTH, extract_depth: str = EXTRACT_DEPTH
):
    """Crawl documentation site using Tavily."""
    log_info(
        "🗺️  TavilyCrawl: Starting to crawl the documentation site",
        Colors.PURPLE,
    )

    tavily_tools = get_tavily_tools()
    tavily_crawl = tavily_tools["crawl"]

    res = tavily_crawl.invoke(
        {"url": url, "max_depth": max_depth, "extract_depth": extract_depth}
    )

    return res["results"]


# Main Orchestrator
async def main():
    """Main async function to orchestrate the entire process."""
    # Setup SSL
    setup_ssl()

    log_header("DOCUMENTATION INGESTION PIPELINE")

    # Crawl the documentation site
    all_docs = crawl_documentation()

    # Split documents into chunks
    splitted_docs = split_documents(all_docs)

    # Process documents asynchronously
    await index_documents_async(splitted_docs)

    # Log summary
    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


# Entry Point
if __name__ == "__main__":
    asyncio.run(main())
