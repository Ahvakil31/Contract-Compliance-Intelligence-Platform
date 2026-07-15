import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup, AutoTokenizer, AutoConfig
from sklearn.metrics import f1_score
import numpy as np
from dataset_handler import LegalClauseDataset

def train_legal_classifier(train_dataset: LegalClauseDataset, num_labels: int, epochs: int = 20, batch_size: int = 2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Execution platform selected: {device}")

    model_name = "nlpaueb/legal-bert-base-uncased"
    
    id2label = {0: "Auto-Renewal", 1: "Indemnification", 2: "Confidentiality"}
    label2id = {v: k for k, v in id2label.items()}

    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels
    config.problem_type = "multi_label_classification"
    config.id2label = id2label
    config.label2id = label2id

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        config=config
    )
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    
    total_steps = len(train_loader) * epochs
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    print("🚀 Initializing Fine-Tuning Loops...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # FIXED: Explicit float cast to prevent Cross-Entropy Long vs Float runtime crashes
            labels = batch["labels"].to(device).float()
            
            # FIXED: Slice labels down to num_labels dimension if padded by dataset handler
            if labels.shape[-1] != num_labels:
                labels = labels[:, :num_labels]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            probs = torch.sigmoid(logits).cpu().detach().numpy()
            all_preds.append(probs)
            all_targets.append(labels.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        
        predictions_matrix = np.vstack(all_preds) >= 0.5
        targets_matrix = np.vstack(all_targets)
        macro_f1 = f1_score(targets_matrix, predictions_matrix, average="macro", zero_division=0)
        
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"📊 Epoch {epoch+1:02d}/{epochs:02d} — Loss: {avg_loss:.4f} — Macro F1-Score: {macro_f1:.4f}")

    # Export weights alongside full tokenizer configurations for clean tasks.py compatibility
    model.save_pretrained("./fine_tuned_legal_classifier")
    train_dataset.tokenizer.save_pretrained("./fine_tuned_legal_classifier")
    print("🎯 Model and Tokenizer artifacts successfully exported to './fine_tuned_legal_classifier'")

if __name__ == "__main__":
    mock_texts = [
        "This Agreement will automatically renew for subsequent 12-month periods unless terminated by notice.",
        "The Supplier shall indemnify and defend the Buyer against any intellectual property infringement claims.",
        "Neither party shall disclose confidential trade secrets without explicit prior written authorization."
    ]
    
    mock_labels = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]

    print("⏳ Building tokenization data vectors...")
    ds = LegalClauseDataset(texts=mock_texts, labels=mock_labels, model_name="nlpaueb/legal-bert-base-uncased")
    
    # Run the classification loop
    train_legal_classifier(train_dataset=ds, num_labels=3, epochs=20, batch_size=2)