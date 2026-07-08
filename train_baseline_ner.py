import os
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random
from spacy.tokens import DocBin

def train_custom_baseline_ner(train_data_path: str, output_dir: str, iterations: int = 10):
    """Programmatically instantiates an NER component and executes a localized loop optimization."""
    # Step 1: Construct an empty English language model
    nlp = spacy.blank("en")
    
    # Step 2: Add an NER component pipeline step if missing
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Load documentation bins back to unpack exact Example objects
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"❌ Target training binary file not found at: '{train_data_path}'. Please run ner_formatter.py first.")

    doc_bin = DocBin().from_disk(train_data_path)
    reference_docs = list(doc_bin.get_docs(nlp.vocab))
    
    training_examples = []
    for doc in reference_docs:
        # Register explicit labels found inside training assets
        for ent in doc.ents:
            ner.add_label(ent.label_)
        # Create an execution pair holding predicted vs reference structures
        training_examples.append(
            Example.from_dict(
                nlp.make_doc(doc.text), 
                {"entities": [(e.start_char, e.end_char, e.label_) for e in doc.ents]}
            )
        )

    # Step 3: Begin Training with selective pipe component insulation
    print("🚀 Initializing pipeline optimizers...")
    optimizer = nlp.begin_training()
    
    # Isolate training context strictly to the NER component
    with nlp.select_pipes(enable="ner"):
        for epoch in range(iterations):
            random.shuffle(training_examples)
            losses = {}
            
            # Batch sizes compound exponentially from 2 up to 8 for fast evaluation steps
            batches = minibatch(training_examples, size=compounding(2.0, 8.0, 1.001))
            for batch in batches:
                nlp.update(
                    batch,
                    drop=0.2,      # 20% Dropout rate prevents early model overfitting
                    sgd=optimizer, # Stochastic Gradient Descent tracking bounds
                    losses=losses,
                )
            print(f"📊 Epoch {epoch+1:02d}/{iterations:02d} — Loss Factor: {losses['ner']:.4f}")

    # Save out the structural model artifacts
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    nlp.to_disk(output_dir)
    print(f"🎯 Model output successfully exported to '{output_dir}'")

# Execution Wrapper
if __name__ == "__main__":
    # FIXED: Updated target source file to mirror 'legal_ner_train.spacy' from ner_formatter.py
    TARGET_TRAIN_DATA = "legal_ner_train.spacy"
    OUTPUT_MODEL_DIR = "./legal_ner_model"
    
    # Run the optimized training loop
    train_custom_baseline_ner(TARGET_TRAIN_DATA, OUTPUT_MODEL_DIR, iterations=50)
    
    # Verify the baseline output directly
    print("\n🧐 Verification Check on Newly Trained Extraction Baseline:")
    
    try:
        nlp_inference = spacy.load(OUTPUT_MODEL_DIR)
        print("   ↳ Pass: Saved model package loaded cleanly into memory.")
        
        ner_pipe = nlp_inference.get_pipe("ner")
        print(f"   ↳ Registered Target Labels: {ner_pipe.labels}")
        
        test_text = "This Lease Agreement is entered into on June 18, 2026 by Acme Corporation."
        test_doc = nlp_inference(test_text)
        
        print(f"   ↳ Processing test prompt: '{test_text}'")
        if not test_doc.ents:
            print("   ↳ 🔍 Inference Result: No entities isolated. (Need more epochs or training variations)")
        else:
            for ent in test_doc.ents:
                print(f"   ↳ Extracted Entity: [{ent.text}] -> Found Category Label: {ent.label_}")
                
    except Exception as e:
        print(f"   ↳ ❌ Verification Check Failed: {e}")