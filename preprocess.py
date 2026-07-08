import json
import torch
from typing import List, Dict
from transformers import AutoTokenizer
import spacy
from document_Schemas import CUADDocument, CUADAnnotation

class LegalDataPreprocessor:
    def __init__(self, model_name: str = "roberta-base"):
        # Load the fast tokenizer variant required for char-to-token mapping
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    def load_cuad_json(self, file_path: str) -> List[CUADDocument]:
        """Parses raw CUAD data into structured Pydantic models."""
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        documents = []
        # CUAD typically follows standard SQuAD-like JSON structure
        for data_item in raw_data["data"]:
            for paragraph in data_item["paragraphs"]:
                context = paragraph["context"]
                title = data_item["title"]
                annotations_dict = {}
                
                for qas in paragraph["qas"]:
                    clause_type = qas["id"]  # e.g., "Limitation of Liability"
                    annotations_dict[clause_type] = []
                    
                    for answer in qas["answers"]:
                        annotations_dict[clause_type].append(
                            CUADAnnotation(
                                start=answer["answer_start"],
                                end=answer["answer_start"] + len(answer["text"]),
                                text=answer["text"]
                            )
                        )
                
                documents.append(
                    CUADDocument(title=title, context=context, annotations=annotations_dict)
                )
        return documents

    def tokenize_and_align(self, doc: CUADDocument, target_clause: str) -> Dict[str, List[int]]:
        """Tokenizes text and aligns character-level annotations with sub-token indices."""
        # Use spaCy for string cleanup if necessary, then tokenize with Hugging Face
        tokenized_inputs = self.tokenizer(
            doc.context,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True
        )

        labels = [0] * len(tokenized_inputs["input_ids"])
        offset_mapping = tokenized_inputs["offset_mapping"]
        target_annotations = doc.annotations.get(target_clause, [])

        for anon in target_annotations:
            start_char = anon.start
            end_char = anon.end

            for idx, (token_start, token_end) in enumerate(offset_mapping):
                # Skip special tokens (e.g., CLS, SEP) which have (0,0) offsets
                if token_start == 0 and token_end == 0:
                    continue
                
                # If the token falls cleanly inside the annotated character window
                if token_start >= start_char and token_end <= end_char:
                    labels[idx] = 1  # Token belongs to target clause

        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": labels
        }

# Execution wrapper for Day 1-2 pipeline testing
if __name__ == "__main__":
    preprocessor = LegalDataPreprocessor("roberta-base")
    
    # Mock data showing structural match to CUAD formatting
    mock_cuad_data = {
        "data": [{
            "title": "Alpha_Beta_Agreement_2026",
            "paragraphs": [{
                "context": "This Agreement shall be governed by the laws of the State of New York.",
                "qas": [{
                    "id": "Governing Law",
                    "answers": [{"answer_start": 40, "text": "the State of New York"}]
                }]
            }]
        }]
    }
    
    with open("mock_cuad.json", "w") as f:
        json.dump(mock_cuad_data, f)
        
    print("⏳ Processing raw data assets...")
    docs = preprocessor.load_cuad_json("mock_cuad.json")
    print(f"✅ Successfully verified and parsed {len(docs)} document structures.")
    
    # Align token arrays
    features = preprocessor.tokenize_and_align(docs[0], target_clause="Governing Law")
    print("✅ Token alignment complete.")
    print(f"Input IDs (Truncated): {features['input_ids'][:15]}")
    print(f"Aligned Labels (Truncated): {features['labels'][:15]}")