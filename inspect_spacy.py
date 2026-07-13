import spacy
from spacy.tokens import DocBin

# 1. Initialize a blank language model matching your training data (e.g., English)
nlp = spacy.blank("en")

# 2. Load the binary .spacy file
doc_bin = DocBin().from_disk("legal_ner_train.spacy")

# 3. Extract and print the documents and their named entities (NER)
docs = list(doc_bin.get_docs(nlp.vocab))

print(f"📊 Total documents found: {len(docs)}\n")
print("--- Sample Entities Summary ---")

# Inspect the first 5 parsed documents
for i, doc in enumerate(docs[:5]):
    print(f"\n📄 Document #{i + 1}:")
    print(f"Text: {doc.text[:150]}...")  # showing truncated text
    
    if doc.ents:
        print("Entities:")
        for ent in doc.ents:
            print(f"  - [{ent.label_}] -> '{ent.text}' (Indices: {ent.start_char}:{ent.end_char})")
    else:
        print("Entities: None found in this document.")