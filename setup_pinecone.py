"""
Setup script to create Pinecone index for the documentation helper.
Run this before running ingestion.py for the first time.
"""
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

def create_pinecone_index():
    """Create a Pinecone index with the correct configuration."""

    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    index_name = os.getenv("PINECONE_INDEX_NAME", "dochelp")

    # Check if index already exists
    existing_indexes = pc.list_indexes()
    index_names = [idx.name for idx in existing_indexes]

    if index_name in index_names:
        print(f"✅ Index '{index_name}' already exists!")
        return

    print(f"🔨 Creating new Pinecone index: {index_name}")

    # Create index with serverless configuration
    # Using dimension 1536 for OpenAI text-embedding-3-small model
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI text-embedding-3-small dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # You can change this to your preferred region
        )
    )

    print(f"✅ Successfully created index '{index_name}'!")
    print("📝 You can now run ingestion.py to populate the index with data.")

if __name__ == "__main__":
    create_pinecone_index()