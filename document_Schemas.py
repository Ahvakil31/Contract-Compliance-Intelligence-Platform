from pydantic import BaseModel, Field
from typing import List, Dict, Optional , Any

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