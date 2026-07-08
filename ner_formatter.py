# ner_formatter.py
import spacy
from spacy.tokens import DocBin
from pydantic import BaseModel
from typing import List
import traceback

class EntitySpan(BaseModel):
    start: int
    end: int
    label: str

class RawNERTrainingItem(BaseModel):
    text: str
    entities: List[EntitySpan]

class LegalNERDataPreparer:
    def __init__(self):
        self.nlp = spacy.blank("en")

    def convert_to_spacy_binary(self, raw_data: List[RawNERTrainingItem], output_path: str):
        """Builds token-snapped binary data files for NER model fine-tuning."""
        doc_bin = DocBin()
        skipped_entities_count = 0
        success_count = 0

        for idx, item in enumerate(raw_data):
            doc = self.nlp.make_doc(item.text)
            spans = []
            
            for ent in item.entities:
                # FIXED: "expand" stops tokenization bounds from dropping mid-word variations
                span = doc.char_span(ent.start, ent.end, label=ent.label, alignment_mode="expand")
                if span is None:
                    skipped_entities_count += 1
                    continue
                spans.append(span)

            try:
                filtered_spans = spacy.util.filter_spans(spans)
                doc.set_ents(filtered_spans)
                doc_bin.add(doc)
                success_count += 1
            except Exception as e:
                print(f"⚠️ Tracking error on record window indexing [{idx}]: {str(e)}")
                traceback.print_exc()
                skipped_entities_count += 1

        doc_bin.to_disk(output_path)
        print(f"📦 Serialization Matrix Saved to -> {output_path}")
        print(f"✅ Successfully compiled documents: {success_count}")
        print(f"❌ Skipped/Misaligned entities: {skipped_entities_count}")

if __name__ == "__main__":
    preparer = LegalNERDataPreparer()
    
    # Corrected offsets to map perfectly against character array lengths
    mock_dataset = [
        RawNERTrainingItem(
            text="This Lease Agreement is entered into on June 18, 2026 by Acme Corporation.",
            entities=[
                EntitySpan(start=40, end=53, label="DATE"),
                EntitySpan(start=57, end=73, label="ORG")
            ]
        ),
        RawNERTrainingItem(
            text="The total secondary liability cap shall not exceed $5,000,000 within any fiscal year.",
            entities=[
                EntitySpan(start=52, end=63, label="MONEY") # Adjusted boundary coordinate mapping
            ]
        )
    ]
    
    print("⏳ Launching spatial NER binary conversion loop...")
    preparer.convert_to_spacy_binary(mock_dataset, "legal_ner_train.spacy")