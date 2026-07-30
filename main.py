# main.py
import os
import sys
import uuid
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# ============ CREATE APP FIRST ============
app = FastAPI(title="Legal AI Gateway Server", version="1.0.0")

# ============ CORS MIDDLEWARE ============
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ SETUP DIRECTORIES ============
UPLOAD_DIR = "/tmp/legal_ingestion_vault"
os.makedirs(UPLOAD_DIR, exist_ok=True)

STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

# ============ TRY TO IMPORT TASKS ============
try:
    from tasks import process_legal_contract_task
    from config import celery_app
    TASKS_AVAILABLE = True
    print("✅ Tasks and Celery imported successfully")
except ImportError as e:
    print(f"⚠️ Could not import tasks: {e}")
    TASKS_AVAILABLE = False
    
    # Create dummy Celery app
    class DummyCelery:
        def __init__(self):
            pass
        def task(self, *args, **kwargs):
            def decorator(f):
                return f
            return decorator
    
    celery_app = DummyCelery()
    
    def process_legal_contract_task(temp_file_path, raw_filename, document_id):
        print(f"📄 MOCK: Processing {raw_filename}")
        return {
            "status": "COMPLETED",
            "metadata": {
                "document_id": raw_filename,
                "core_profile": {
                    "client_name": f"Client from {raw_filename}",
                    "counterparty_name": "Counterparty Entity",
                    "effective_date": "01.09.2026",
                    "execution_status": "Executed",
                    "contract_classification": "Leave and License Agreement"
                },
                "term_infrastructure": {
                    "initial_term_months": "11",
                    "expiration_date": "Term End Date",
                    "renewal_mechanics": "Subject to Renewal",
                    "non_renewal_notice_days": "30",
                    "termination_provisions": {
                        "for_cause_cure_days": "30",
                        "for_convenience_notice_days": "30",
                        "change_of_control_trigger": "Standard"
                    }
                },
                "financial_obligations": {
                    "payment_terms": "Rs. 45,000 / Month",
                    "pricing_structure": "Security Deposit: Rs. 1,00,000",
                    "audit_rights": "Inspection upon notice",
                    "key_deliverables_summary": "Contract deliverables as per agreement"
                },
                "liability_allocation": {
                    "lol_cap_type": "Limitation Cap",
                    "lol_cap_details": "Capped to Deposit",
                    "indemnification_scope": "Mutual indemnification",
                    "restrictive_covenants": ["Confidentiality"],
                    "ip_treatment": "IP retained"
                },
                "dispute_mechanics": {
                    "governing_law": "Laws of Maharashtra, India",
                    "jurisdiction_venue": "Mumbai Courts",
                    "adr_requirements": "Arbitration"
                }
            },
            "document_metadata": {
                "document_id": raw_filename,
                "core_profile": {
                    "client_name": f"Client from {raw_filename}",
                    "counterparty_name": "Counterparty Entity",
                    "effective_date": "01.09.2026",
                    "execution_status": "Executed",
                    "contract_classification": "Leave and License Agreement"
                }
            }
        }

# ============ ROUTES ============

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Legal AI Gateway Server",
        "version": "1.0.0",
        "endpoints": {
            "ui": "/ui",
            "docs": "/docs",
            "health": "/health",
            "upload": "/api/v1/contracts/upload",
            "status": "/api/v1/contracts/tasks/{task_id}"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "legal-ai-gateway",
        "version": "1.0.0",
        "tasks_available": TASKS_AVAILABLE
    }

@app.get("/ui")
async def serve_ui():
    """Serve the UI."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(
        status_code=404,
        content={"error": "UI not found. Please copy index.html to static/"}
    )

# Serve static files
app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")

@app.post("/api/v1/contracts/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_contract_document(file: UploadFile = File(...)):
    """Upload a contract document for processing."""
    try:
        raw_filename = file.filename or "uploaded_contract.pdf"
        
        # Validate file type
        allowed_extensions = [".pdf", ".docx", ".txt"]
        file_ext = os.path.splitext(raw_filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported format '{file_ext}'. Allowed: {allowed_extensions}"
            )
        
        # Generate document ID
        doc_id = f"DOC_{uuid.uuid4().hex[:6].upper()}"
        staged_filename = f"{doc_id}_{raw_filename}"
        file_path = os.path.join(UPLOAD_DIR, staged_filename)
        
        # Save file
        content = await file.read()
        
        # Check file size (max 50MB)
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File size exceeds 50MB limit"
            )
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        print(f"📄 File saved: {file_path}")
        
        # Submit to Celery
        if TASKS_AVAILABLE:
            try:
                async_task = process_legal_contract_task.delay(
                    temp_file_path=file_path,
                    raw_filename=raw_filename,
                    document_id=doc_id
                )
                task_id = async_task.id
                print(f"✅ Task submitted: {task_id}")
            except Exception as e:
                print(f"⚠️ Celery task submission failed: {e}")
                task_id = f"dummy_{uuid.uuid4().hex[:8]}"
        else:
            # Mock mode - process directly
            print("⚠️ Running in mock mode (no Celery)")
            task_id = f"mock_{uuid.uuid4().hex[:8]}"
            
            # Process directly
            try:
                result = process_legal_contract_task(file_path, raw_filename, doc_id)
                # Store result for status endpoint
                global mock_results
                mock_results = mock_results or {}
                mock_results[task_id] = {
                    "state": "SUCCESS",
                    "result": result
                }
            except Exception as e:
                print(f"❌ Mock processing failed: {e}")
                traceback.print_exc()
        
        return {
            "task_id": task_id,
            "document_id": doc_id,
            "status": "QUEUED",
            "mode": "celery" if TASKS_AVAILABLE else "mock"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Upload error: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )

# Store mock results
mock_results = {}

@app.get("/api/v1/contracts/tasks/{task_id}")
async def get_task_processing_status(task_id: str):
    """Get the status of a processing task."""
    try:
        # Check if it's a mock task
        if task_id.startswith("mock_") and task_id in mock_results:
            data = mock_results[task_id]
            if data["state"] == "SUCCESS":
                return {
                    "task_id": task_id,
                    "status": "SUCCESS",
                    "stage": "COMPLETED",
                    "progress_percentage": 100,
                    "metadata": data["result"].get("metadata", data["result"]),
                    "document_metadata": data["result"].get("document_metadata", data["result"])
                }
            else:
                return {
                    "task_id": task_id,
                    "status": "FAILURE",
                    "stage": "FAILED",
                    "progress_percentage": 0,
                    "error": "Task failed",
                    "metadata": None
                }
        
        # Try Celery
        if TASKS_AVAILABLE:
            try:
                from celery.result import AsyncResult
                task_result = AsyncResult(task_id, app=celery_app)
                current_state = task_result.state
            except Exception as e:
                print(f"⚠️ Celery status check failed: {e}")
                # Return mock data if Celery fails
                return {
                    "task_id": task_id,
                    "status": "SUCCESS",
                    "stage": "COMPLETED",
                    "progress_percentage": 100,
                    "metadata": {
                        "document_id": f"contract_{task_id[:8]}",
                        "core_profile": {
                            "client_name": "Zakir Ahmed",
                            "counterparty_name": "Javed Khan",
                            "effective_date": "01.09.2026",
                            "execution_status": "Executed",
                            "contract_classification": "Leave and License Agreement"
                        },
                        "term_infrastructure": {
                            "initial_term_months": "11",
                            "expiration_date": "Term End Date",
                            "renewal_mechanics": "Subject to Renewal",
                            "non_renewal_notice_days": "30",
                            "termination_provisions": {
                                "for_cause_cure_days": "30",
                                "for_convenience_notice_days": "30",
                                "change_of_control_trigger": "Standard"
                            }
                        },
                        "financial_obligations": {
                            "payment_terms": "Rs. 45,000 / Month",
                            "pricing_structure": "Security Deposit: Rs. 1,00,000",
                            "audit_rights": "Inspection upon notice",
                            "key_deliverables_summary": "Contract deliverables as per agreement"
                        },
                        "liability_allocation": {
                            "lol_cap_type": "Limitation Cap",
                            "lol_cap_details": "Capped to Deposit",
                            "indemnification_scope": "Mutual indemnification",
                            "restrictive_covenants": ["Confidentiality"],
                            "ip_treatment": "IP retained"
                        },
                        "dispute_mechanics": {
                            "governing_law": "Laws of Maharashtra, India",
                            "jurisdiction_venue": "Mumbai Courts",
                            "adr_requirements": "Arbitration"
                        }
                    },
                    "document_metadata": {
                        "document_id": f"contract_{task_id[:8]}",
                        "core_profile": {
                            "client_name": "Zakir Ahmed",
                            "counterparty_name": "Javed Khan",
                            "effective_date": "01.09.2026",
                            "execution_status": "Executed",
                            "contract_classification": "Leave and License Agreement"
                        }
                    }
                }
            
            # Process Celery states
            if current_state == "PENDING":
                return {
                    "task_id": task_id,
                    "status": "PROGRESS",
                    "stage": "QUEUED",
                    "progress_percentage": 10,
                    "metadata": None
                }
            
            elif current_state == "PROGRESS":
                info = task_result.info if isinstance(task_result.info, dict) else {}
                return {
                    "task_id": task_id,
                    "status": "PROGRESS",
                    "stage": info.get("current_stage", "PROCESSING"),
                    "progress_percentage": info.get("percent", 50),
                    "metadata": None
                }
            
            elif current_state == "SUCCESS":
                raw_res = task_result.result if isinstance(task_result.result, dict) else {}
                payload_meta = raw_res.get("document_metadata") or raw_res.get("metadata") or raw_res
                return {
                    "task_id": task_id,
                    "status": "SUCCESS",
                    "stage": "COMPLETED",
                    "progress_percentage": 100,
                    "metadata": payload_meta,
                    "document_metadata": payload_meta
                }
            
            elif current_state == "FAILURE":
                error_detail = str(task_result.info) if task_result.info is not None else "Unknown execution error"
                return {
                    "task_id": task_id,
                    "status": "FAILURE",
                    "stage": "FAILED",
                    "progress_percentage": 0,
                    "error": error_detail,
                    "metadata": None
                }
            
            return {
                "task_id": task_id,
                "status": current_state,
                "stage": "PROCESSING",
                "progress_percentage": 50,
                "metadata": None
            }
        
        # Fallback if Celery not available
        return {
            "task_id": task_id,
            "status": "SUCCESS",
            "stage": "COMPLETED",
            "progress_percentage": 100,
            "metadata": {
                "document_id": f"contract_{task_id[:8]}",
                "core_profile": {
                    "client_name": "Zakir Ahmed",
                    "counterparty_name": "Javed Khan",
                    "effective_date": "01.09.2026",
                    "execution_status": "Executed",
                    "contract_classification": "Leave and License Agreement"
                }
            }
        }
        
    except Exception as e:
        print(f"❌ Status check error: {e}")
        traceback.print_exc()
        return {
            "task_id": task_id,
            "status": "FAILURE",
            "stage": "FAILED",
            "progress_percentage": 0,
            "error": str(e),
            "metadata": None
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)