#!/usr/bin/env python3

import random
import itertools
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaForMaskedLM, XLMRobertaTokenizer
from datasets import load_dataset

class MultilingualDataset(Dataset):
    def __init__(self, texts, language_labels, tokenizer, max_length=128):
        self.texts = texts
        self.language_labels = language_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        lang = self.language_labels[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["language"] = lang
        return enc

def augment_text(text, drop_prob=0.1):
    tokens = text.split()
    return " ".join(t for t in tokens if random.random() > drop_prob)

def compute_fairness_loss(hidden_states, language_ids):
    sent_emb = hidden_states.mean(dim=1)  
    langs = torch.unique(language_ids)
    if langs.numel() < 2:
        return torch.tensor(0.0, device=hidden_states.device)
    means = []
    for lang in langs:
        mask = (language_ids == lang)
        if mask.sum() > 0:
            means.append(sent_emb[mask].mean(dim=0))
    if len(means) < 2:
        return torch.tensor(0.0, device=hidden_states.device)
    group_means = torch.stack(means)  
    return group_means.var(dim=0, unbiased=False).mean()

def train_model(model, dataloader, optimizer, device,
                num_epochs=3, fairness_weight=0.1):
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone().to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True
            )
            mlm_loss = outputs.loss
            hidden_states = outputs.hidden_states[-1]
            language_ids = torch.tensor(batch["language"]).to(device)
            fairness_loss = compute_fairness_loss(hidden_states, language_ids)

            total_loss = mlm_loss + fairness_weight * fairness_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            print(
                f"Epoch {epoch+1}: MLM={mlm_loss.item():.4f}, "
                f"Fairness={fairness_loss.item():.4f}, "
                f"Total={total_loss.item():.4f}"
            )

def load_text_file(file_path, label):
    """Load text data from a file and assign a label."""
    texts, labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            texts.append(line.strip())
            labels.append(label)
    return texts, labels

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "xlm-roberta-base"

    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    model = XLMRobertaForMaskedLM.from_pretrained(
        model_name, output_hidden_states=True
    ).to(device)

    eng_dev_texts, eng_dev_labels = load_text_file(
        "/Users/evelyn.e/Desktop/Spring2025/AI/train_Multilingual/flores101_dataset/dev/eng.dev", label=0
    )
    swh_dev_texts, swh_dev_labels = load_text_file(
        "/Users/evelyn.e/Desktop/Spring2025/AI/train_Multilingual/flores101_dataset/dev/swh.dev", label=1
    )
    eng_devtest_texts, eng_devtest_labels = load_text_file(
        "/Users/evelyn.e/Desktop/Spring2025/AI/train_Multilingual/flores101_dataset/devtest/eng.devtest", label=0
    )
    swh_devtest_texts, swh_devtest_labels = load_text_file(
        "/Users/evelyn.e/Desktop/Spring2025/AI/train_Multilingual/flores101_dataset/devtest/swh.devtest", label=1
    )

    texts = eng_dev_texts + swh_dev_texts + eng_devtest_texts + swh_devtest_texts
    labels = eng_dev_labels + swh_dev_labels + eng_devtest_labels + swh_devtest_labels

    aug_texts, aug_labels = [], []
    for t, lang in zip(texts, labels):
        aug_texts.append(t)
        aug_labels.append(lang)
        if lang == 1 and random.random() > 0.5:  
            aug_texts.append(augment_text(t))
            aug_labels.append(lang)

    dataset = MultilingualDataset(aug_texts, aug_labels, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    train_model(
        model, dataloader, optimizer, device,
        num_epochs=3, fairness_weight=0.1
    )
