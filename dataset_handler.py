# dataset_handler.py
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Union

class LegalClauseDataset(Dataset):
    def __init__(
        self, 
        texts: List[str], 
        labels: List[List[Union[int, float]]], 
        model_name: str = "nlpaueb/legal-bert-base-uncased", 
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            # Target needs to match class count (e.g., shape [3]) and be torch.float for BCE loss
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }