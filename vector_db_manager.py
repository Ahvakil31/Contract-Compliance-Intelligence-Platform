import os
import time
from typing import List, Any, Optional
from langchain_huggingface import HuggingFaceEmbeddings
# Use the updated, non-colliding LangChain wrapper class name
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from chunking_engine import LegalChunkingPipeline, IngestedParagraph

class LegalVectorDBManager:
    def __init__(self, index_name: str = "legal-compliance-vault"):
        self.index_name = index_name
        
        # Initialize an optimized embedding model generating 768-dimensional dense vectors
        print("⏳ Loading Transformer Embedding Architecture...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5", # High-performance public 768-dimension embedding engine
            model_kwargs={'device': 'cpu'} # Switch to 'cuda' if hardware accelerator is available
        )
        
        # Check if running in mock/local mode
        self.api_key = os.environ.get("PINECONE_API_KEY", "dummy_pinecone_key_for_local_testing")
        self.is_mock_mode = self.api_key in ["dummy_pinecone_key_for_local_testing", "mock-key-for-compilation"]
        
        # Initialize Pinecone Client
        self.pc = Pinecone(api_key=self.api_key)

    def setup_index(self):
        """Creates the serverless vector database index if it does not already exist."""
        if self.is_mock_mode:
            print("⚠️ [Mock Mode] Skipping live Pinecone index creation check.")
            return

        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"🏗️ Index missing. Creating new Pinecone index: '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=768, # Matches the exact output dimensions of bge-base-en-v1.5
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            
            # FIXED: Converted bracket dictionary lookup to dot attribute lookup to prevent subscriptable crashes
            print("⏳ Waiting for Pinecone cloud index allocation to spin up...")
            while not self.pc.describe_index(self.index_name).status.ready:
                time.sleep(2)
            print("🚀 Index is live and ready!")
        else:
            print(f"✅ Active Vector Index Found: '{self.index_name}'")

    def populate_database(self, documents: List[Document]) -> Any:
        """Embeds text chunks and uploads vectors alongside metadata packages."""
        if self.is_mock_mode:
            print(f"⚠️ [Mock Mode] Skipping vector streaming for {len(documents)} documents to Pinecone cloud.")
            return None

        self.setup_index()
        
        print(f"🚀 Streaming {len(documents)} vectors to Pinecone cloud storage cluster...")
        vector_store = PineconeVectorStore.from_documents(
            documents, 
            self.embeddings, 
            index_name=self.index_name
        )
        print("🎉 Database population execution successfully complete.")
        return vector_store

    def semantic_query(
        self, 
        query: str, 
        top_k: int = 3, 
        pillar_filter: Optional[str] = None,
        field_filter: Optional[str] = None
    ) -> List[Document]:
        """
        Queries database using semantic vector matching with enterprise schema metadata filters.
        Allows downstream processing nodes to pinpoint contexts isolated by operational pillars.
        """
        # Construct Pinecone metadata filtering conditions matching the updated chunk keys
        filter_dict = {}
        if pillar_filter:
            filter_dict["pillar_category"] = pillar_filter
        if field_filter:
            filter_dict["specific_field"] = field_filter

        print(f"🔍 Searching for context vectors matching query: '{query}' | Filters: {filter_dict}")

        if self.is_mock_mode:
            print("⚠️ [Mock Mode] Connection intercepted safely during offline execution.")
            return []

        vector_store = PineconeVectorStore.from_existing_index(self.index_name, self.embeddings)
        results = vector_store.similarity_search(query, k=top_k, filter=filter_dict if filter_dict else None)
        return results

# Functional Mock Ingestion Verification Loop
if __name__ == "__main__":
    # Set explicit dummy values to pass structural assertions inside LangChain modules
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    # Assembly Pipeline Workflow Test using updated pillar chunk layouts
    chunker = LegalChunkingPipeline()
    db_manager = LegalVectorDBManager()
    
    sample_source = [
        IngestedParagraph(
            text="Governing law shall be interpreted under the statutes of New York state courts.",
            page_number=12,
            document_id="CONTRACT_MOCK_01",
            pillar_category="term_infrastructure",
            specific_field="governing_law"
        )
    ]
    
    docs = chunker.generate_langchain_docs(sample_source)
    print("📋 Document processing verification completed safely.")
    
    # Verify semantic query filtering layout locally
    print("\n🔬 Validating metadata filter routing interface:")
    db_manager.semantic_query(
        query="Which state's courts govern this agreement?", 
        pillar_filter="term_infrastructure",
        field_filter="governing_law"
    )