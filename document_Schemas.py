from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

# ======================================================================
# 📄 OCR & Annotation Schemas
# ======================================================================

class CUADAnnotation(BaseModel):
    start: int
    end: int
    text: str

class CUADDocument(BaseModel):
    title: str
    context: str
    # Maps clause type (e.g., "Governing Law") to its character annotations
    annotations: Dict[str, List[CUADAnnotation]]

class TokenizedOutput(BaseModel):
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]  # Token-level BIO tags or binary sequence labels

class BoundingBox(BaseModel):
    x_min: int = Field(..., description="Top-left X coordinate")
    y_min: int = Field(..., description="Top-left Y coordinate")
    x_max: int = Field(..., description="Bottom-right X coordinate")
    y_max: int = Field(..., description="Bottom-right Y coordinate")

class OCRWord(BaseModel):
    text: str
    confidence: float
    bbox: BoundingBox

class OCRParagraph(BaseModel):
    text: str
    confidence: float
    words: List[OCRWord]

class ProcessedPage(BaseModel):
    page_number: int
    dimensions: tuple = Field(..., description="(Width, Height) of the page image")
    text_content: str
    paragraphs: List[OCRParagraph]
    is_scanned: bool


# ======================================================================
# 🏛️ Legal Analysis Pillar Schemas (Required by tasks.py)
# ======================================================================

class CoreProfileMetadata(BaseModel):
    client_name: str = Field(..., description="Name of the client entity")
    counterparty_name: str = Field(..., description="Name of the counterparty entity")
    effective_date: str = Field(..., description="Effective date of agreement execution")
    execution_status: str = Field(..., description="Execution state (e.g., Fully Executed, Draft)")
    contract_classification: str = Field(..., description="Classification category of agreement")

class TerminationMechanics(BaseModel):
    for_cause_cure_days: int = Field(..., description="Cure period allowance in days")
    for_convenience_notice_days: int = Field(..., description="Termination notice window in days")
    change_of_control_trigger: str = Field(..., description="Corporate structural change trigger state")

class TermExitInfrastructure(BaseModel):
    initial_term_months: int = Field(..., description="Initial term duration in months")
    expiration_date: str = Field(..., description="Expiration date of contract")
    renewal_mechanics: str = Field(..., description="Renewal behavior (e.g., Evergreen, Fixed Term)")
    non_renewal_notice_days: int = Field(..., description="Notice window to prevent auto-renewal")
    termination_provisions: TerminationMechanics

class FinancialOperationalObligations(BaseModel):
    payment_terms: str = Field(..., description="Payment billing terms (e.g., Net 30)")
    pricing_structure: str = Field(..., description="Fee and pricing structure terms")
    audit_rights: str = Field(..., description="Audit and records inspection provisions")
    key_deliverables_summary: str = Field(..., description="Summary of core deliverables and scope")

class LegalRiskLiabilityAllocation(BaseModel):
    lol_cap_type: str = Field(..., description="Limitation of Liability cap mechanism")
    lol_cap_details: str = Field(..., description="Detailed description of liability limits")
    indemnification_scope: str = Field(..., description="Indemnification obligations coverage")
    restrictive_covenants: List[str] = Field(default_factory=list, description="List of active covenants")
    ip_treatment: str = Field(..., description="Intellectual Property rights and assignments")

class DisputeVenueMechanics(BaseModel):
    governing_law: str = Field(..., description="Governing state or national law")
    jurisdiction_venue: str = Field(..., description="Court jurisdiction and venue location")
    adr_requirements: str = Field(..., description="Alternative Dispute Resolution provisions")


# ======================================================================
# 🎯 Enterprise Contract Analysis Master Object
# ======================================================================

class EnterpriseContractAnalysis(BaseModel):
    document_id: str = Field(..., description="Unique identifier for the contract")
    core_profile: CoreProfileMetadata
    term_infrastructure: TermExitInfrastructure
    financial_obligations: FinancialOperationalObligations
    liability_allocation: LegalRiskLiabilityAllocation
    dispute_mechanics: DisputeVenueMechanics