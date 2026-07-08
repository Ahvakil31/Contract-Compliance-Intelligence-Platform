# dataset_handler.py
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict

class LegalClauseDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[List[int]], model_name: str = "nlpaueb/legal-bert-base-uncased", max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        raw_labels = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # FIXED: Handles padding and truncation variations coming from token processors
        if len(raw_labels) != self.max_length:
            if len(raw_labels) > self.max_length:
                processed_labels = raw_labels[:self.max_length]
            else:
                processed_labels = raw_labels + [-100] * (self.max_length - len(raw_labels))
        else:
            processed_labels = raw_labels

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(processed_labels, dtype=torch.long)
        }