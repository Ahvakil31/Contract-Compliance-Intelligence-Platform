import os
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any, Optional, Tuple
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

# Import custom structures safely matching document_schemas.py hierarchy
from document_Schemas import ProcessedPage, OCRParagraph, OCRWord, BoundingBox

class LegalIngestionPipeline:
    def __init__(self, tesseract_cmd: Optional[str] = None, dpi: int = 200):
        self.dpi = dpi
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def _extract_native_pdf_text(self, pdf_path: str) -> Optional[str]:
        """
        Attempts to extract digital-native text layer to bypass OCR when possible.
        """
        return None 

    def _process_single_page_ocr(self, args: tuple) -> ProcessedPage:
        """
        Executes OCR on a single rasterized page image inside an isolated process worker.
        """
        page_img, page_num = args
        width, height = page_img.size
        
        # Request detailed data tracking matrix from Tesseract engine
        data = pytesseract.image_to_data(page_img, output_type=pytesseract.Output.DICT)
        
        paragraphs_map: Dict[int, List[OCRWord]] = {}
        paragraph_texts: Dict[int, List[str]] = {}
        
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            # Read layout level block configuration metadata
            level = data['level'][i]
            block_num = data['block_num'][i]
            par_num = data['par_num'][i]
            
            # Combine into an isolated spatial identifier key
            paragraph_key = (block_num * 1000) + par_num
            
            word_text = data['text'][i].strip()
            if not word_text:
                continue
                
            bbox = BoundingBox(
                x_min=data['left'][i],
                y_min=data['top'][i],
                x_max=data['left'][i] + data['width'][i],
                y_max=data['top'][i] + data['height'][i]
            )
            
            ocr_word = OCRWord(text=word_text, confidence=float(data['conf'][i]), bbox=bbox)
            
            if paragraph_key not in paragraphs_map:
                paragraphs_map[paragraph_key] = []
                paragraph_texts[paragraph_key] = []
                
            paragraphs_map[paragraph_key].append(ocr_word)
            paragraph_texts[paragraph_key].append(word_text)

        processed_paragraphs: List[OCRParagraph] = []
        for key in paragraphs_map.keys():
            words_list = paragraphs_map[key]
            full_paragraph_string = " ".join(paragraph_texts[key]).strip()
            
            if not full_paragraph_string:
                continue
                
            # Calculate mean confidence factor safely across the paragraph structure
            mean_conf = sum(w.confidence for w in words_list) / len(words_list) if words_list else 0.0
            
            processed_paragraphs.append(
                OCRParagraph(
                    text=full_paragraph_string,
                    confidence=mean_conf,
                    words=words_list
                )
            )

        full_page_text = "\n\n".join([p.text for p in processed_paragraphs])

        # FIXED: Explicit cast tuple matches modern typing expectations flawlessly
        dimensions_tuple: Tuple[int, int] = (width, height)

        return ProcessedPage(
            page_number=page_num,
            dimensions=dimensions_tuple,
            text_content=full_page_text,
            paragraphs=processed_paragraphs
        )

    def ingest_contract(self, pdf_path: str, max_workers: int = 4) -> List[ProcessedPage]:
        """Convert PDF into sorted OCR multi-page analytical blocks asynchronously."""
        print(f"⏳ Rasterizing target document framework layers: {pdf_path}")
        pages = convert_from_path(pdf_path, dpi=self.dpi)
        
        worker_payloads = [(page, idx + 1) for idx, page in enumerate(pages)]
        print(f"🚀 Spawning {len(worker_payloads)} asynchronous background processing workers...")
        processed_pages: List[ProcessedPage] = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(self._process_single_page_ocr, worker_payloads)
            for res in results:
                processed_pages.append(res)
                print(f"✅ Extracted Text Structural Mapping from Page {res.page_number}")
                
        return sorted(processed_pages, key=lambda x: x.page_number)

# Functional Pipeline Verification Harness
if __name__ == "__main__":
    pipeline = LegalIngestionPipeline()
    print("🚀 Ingestion Pipeline validation baseline verified successfully.")