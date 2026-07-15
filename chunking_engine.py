# chunking_engine.py
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class IngestedParagraph(BaseModel):
    text: str
    page_number: int
    document_id: str
    # Map back to the 5 Pillars of EnterpriseContractAnalysis
    pillar_category: str = Field("Unclassified", description="e.g., core_profile, term_infrastructure, liability_allocation")
    specific_field: Optional[str] = Field(None, description="e.g., indemnification_scope, lol_cap_details")
    # Aligns with tasks.py Stage 2 initialization and Stage 3 inference tracking
    inferred_clause_type: str = Field("Unclassified", description="e.g., Auto-Renewal, Indemnification, Confidentiality")

class LegalChunkingPipeline:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        # Use recursive splitting to break cleanly at paragraph and sentence ends
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def generate_langchain_docs(self, paragraphs: List[IngestedParagraph]) -> List[Document]:
        """Converts target pillar-aligned structured paragraphs into metadata-mapped LangChain Documents."""
        langchain_docs = []
        
        for par in paragraphs:
            # Split paragraph text safely if it exceeds standard chunk size
            chunks = self.splitter.split_text(par.text)
            
            for index, chunk_text in enumerate(chunks):
                # Build metadata mapping matching your master pillar parameters
                metadata = {
                    "document_id": par.document_id,
                    "page_number": par.page_number,
                    "pillar_category": par.pillar_category,
                    "specific_field": par.specific_field or "general",
                    "inferred_clause_type": par.inferred_clause_type,
                    "chunk_index": index
                }
                langchain_docs.append(Document(page_content=chunk_text, metadata=metadata))
                
        print(f"✂️ Processed {len(paragraphs)} structural paragraphs into {len(langchain_docs)} optimized pillar text chunks.")
        return langchain_docs

if __name__ == "__main__":
    # Mock data demonstrating combined classification and structural metadata parameters
    mock_paragraphs = [
        IngestedParagraph(
            text="The Supplier shall indemnify, defend, and hold harmless the Buyer from any third-party intellectual property claims.",
            page_number=4,
            document_id="DOC_XYZ_2026",
            pillar_category="liability_allocation",
            specific_field="indemnification_scope",
            inferred_clause_type="Indemnification"
        )
    ]
    
    pipeline = LegalChunkingPipeline()
    prepared_chunks = pipeline.generate_langchain_docs(mock_paragraphs)
    print("🔍 Sample Chunk Metadata Generated:")
    print(prepared_chunks[0].metadata)