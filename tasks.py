# tasks.py
import os
import re
from typing import Dict, Any
import pytesseract
from pdf2image import convert_from_path
from config import celery_app, SPACY_MODEL
from extraction_engine import ContractExtractor

# --- Dynamic Operating System Binary Bindings ---
if os.name == 'nt':
    try:
        import shutil
        tesseract_path = shutil.which('tesseract')
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    except:
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    POPPLER_BIN_1 = r'C:\Program Files\poppler-26.02.0\Library\bin'
    POPPLER_BIN_2 = r'C:\Program Files\poppler-26.02.0\bin'
    POPPLER_PATH = POPPLER_BIN_1 if os.path.exists(POPPLER_BIN_1) else POPPLER_BIN_2 if os.path.exists(POPPLER_BIN_2) else None
else:
    POPPLER_PATH = None

# Initialize extractor
extractor = ContractExtractor()


def clean_extracted_text(text: str) -> str:
    """Strips PDF binary streams, coordinate noise, and control characters."""
    if not text:
        return ""

    # Remove PDF objects, streams, coordinate pairs
    text = re.sub(r'%\s*<<.*?>>', '', text, flags=re.DOTALL)
    text = re.sub(r'<<.*?>>', '', text, flags=re.DOTALL)
    text = re.sub(r'(-?\d+\.\s*\d+\s*%?)+', ' ', text)
    text = re.sub(r'%PDF-\d\.\d.*?(?=\n|\r)', '', text)
    text = re.sub(r'\d+\s+\d+\s+obj.*?endobj', '', text, flags=re.DOTALL)
    text = re.sub(r'xref\s+\d+\s+\d+.*?(?=trailer|startxref|\n\n)', '', text, flags=re.DOTALL)
    text = re.sub(r'\b(trailer|startxref|EOF|endstream|stream)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\x20-\x7E\n\r]', ' ', text)

    return re.sub(r'\s+', ' ', text).strip()


def extract_pdf_text_robust(file_path: str) -> str:
    """Multi-stage text extractor featuring automatic OCR for scanned PDFs."""
    extracted_text = ""

    # Strategy 1: pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
            extracted_text = " ".join(pages)
            if len(clean_extracted_text(extracted_text)) > 150:
                return extracted_text
    except Exception as e:
        print(f"pdfplumber failed: {e}")

    # Strategy 2: PyMuPDF / fitz
    try:
        import fitz
        doc = fitz.open(file_path)
        pages = [page.get_text() for page in doc]
        extracted_text = " ".join(pages)
        if len(clean_extracted_text(extracted_text)) > 150:
            return extracted_text
    except Exception as e:
        print(f"PyMuPDF failed: {e}")

    # Strategy 3: Tesseract OCR Fallback
    try:
        if POPPLER_PATH:
            images = convert_from_path(file_path, dpi=200, poppler_path=POPPLER_PATH)
        else:
            images = convert_from_path(file_path, dpi=200)

        ocr_pages = [pytesseract.image_to_string(img) for img in images]
        extracted_text = " ".join(ocr_pages)
        if len(clean_extracted_text(extracted_text)) > 50:
            return extracted_text
    except Exception as e:
        print(f"OCR Exception: {e}")

    # Strategy 4: pypdf Fallback
    try:
        import pypdf
        reader = pypdf.PdfReader(file_path)
        pages = [p.extract_text() or "" for p in reader.pages]
        extracted_text = " ".join(pages)
    except Exception as e:
        print(f"pypdf failed: {e}")

    return extracted_text


@celery_app.task(bind=True, name="tasks.process_legal_contract")
def process_legal_contract_task(self, temp_file_path: str, raw_filename: str, document_id: str) -> Dict[str, Any]:
    """Process a legal contract PDF and extract structured data."""
    print(f"📄 Executing pipeline task for: {raw_filename} (Doc ID: {document_id})")
    
    try:
        self.update_state(state="PROGRESS", meta={"current_stage": "TEXT_EXTRACTION", "percent": 20})
        
        # Extract text from PDF
        full_text = ""
        if temp_file_path.lower().endswith(".pdf"):
            full_text = extract_pdf_text_robust(temp_file_path)
        else:
            with open(temp_file_path, "r", encoding="utf-8", errors="ignore") as f:
                full_text = f.read()

        clean_text = clean_extracted_text(full_text)
        
        if len(clean_text) < 100:
            print("⚠️ Warning: Extracted text is very short. OCR may have failed.")
        else:
            print(f"✅ Extracted {len(clean_text)} characters of text")
        
        self.update_state(state="PROGRESS", meta={"current_stage": "ENTITY_EXTRACTION", "percent": 50})
        
        # Extract all contract details using the enhanced extractor
        extracted = extractor.extract_all(clean_text)
        
        # Log extraction results
        print(f"📊 Extraction Results:")
        print(f"  - Client: {extracted['client_name']}")
        print(f"  - Counterparty: {extracted['counterparty_name']}")
        print(f"  - Date: {extracted['effective_date']}")
        print(f"  - Type: {extracted['contract_type']}")
        print(f"  - Term: {extracted['term_duration']}")
        print(f"  - Rent: {extracted['rent']}")
        print(f"  - Deposit: {extracted['deposit']}")
        print(f"  - Law: {extracted['governing_law']}")
        
        # Build the final payload
        payload = {
            "document_id": raw_filename,
            "core_profile": {
                "client_name": extracted["client_name"],
                "counterparty_name": extracted["counterparty_name"],
                "effective_date": extracted["effective_date"],
                "execution_status": extracted["execution_status"],
                "contract_classification": extracted["contract_type"]
            },
            "term_infrastructure": {
                "initial_term_months": extracted["term_duration"],
                "expiration_date": "Term End Date",
                "renewal_mechanics": "Subject to Contract Renewal Terms",
                "non_renewal_notice_days": extracted["notice_period"],
                "termination_provisions": {
                    "for_cause_cure_days": extracted["notice_period"],
                    "for_convenience_notice_days": extracted["notice_period"],
                    "change_of_control_trigger": "Standard Assignment Clause"
                }
            },
            "financial_obligations": {
                "payment_terms": extracted["rent"],
                "pricing_structure": f"Security Deposit: {extracted['deposit']}",
                "audit_rights": "Inspection upon prior notice during business hours",
                "key_deliverables_summary": clean_text[:300] + "..." if len(clean_text) > 300 else clean_text
            },
            "liability_allocation": {
                "lol_cap_type": "Limitation Cap",
                "lol_cap_details": f"Capped to Consideration / Deposit ({extracted['deposit']})",
                "indemnification_scope": "Mutual indemnification for breach, damages & statutory compliance",
                "restrictive_covenants": ["Confidentiality", "Operational Compliance"],
                "ip_treatment": "IP / Title rights retained per contract provisions"
            },
            "dispute_mechanics": {
                "governing_law": extracted["governing_law"],
                "jurisdiction_venue": extracted["jurisdiction"],
                "adr_requirements": extracted["adr"]
            }
        }
        
        self.update_state(state="PROGRESS", meta={"current_stage": "COMPILING_OUTPUT", "percent": 90})
        
        # Clean up temp file
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"✅ Cleaned up temp file: {temp_file_path}")
            except Exception as e:
                print(f"⚠️ Could not remove temp file: {e}")

        return {
            "status": "COMPLETED",
            "metadata": payload,
            "document_metadata": payload
        }

    except Exception as exc:
        print(f"❌ Task failed: {exc}")
        import traceback
        traceback.print_exc()
        
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass
        self.update_state(state="FAILURE", meta={"error_message": str(exc)})
        raise exc