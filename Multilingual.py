import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaForMaskedLM, XLMRobertaTokenizer
import random

class MultilingualDataset(Dataset):
    def __init__(self, texts, language_labels, tokenizer, max_length=128):
        """
        texts: List of input text strings.
        language_labels: List of language IDs (e.g., 0 for English, 1 for Spanish, etc.).
        tokenizer: Pretrained tokenizer instance.
        """
        self.texts = texts
        self.language_labels = language_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        lang = self.language_labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        encoding = {key: value.squeeze(0) for key, value in encoding.items()}
        encoding["language"] = lang  
        return encoding

def augment_text(text, drop_prob=0.1):
    """
    A simple data augmentation function that randomly drops tokens
    to simulate data variability.
    """
    tokens = text.split()
    new_tokens = [token for token in tokens if random.random() > drop_prob]
    return " ".join(new_tokens)


def compute_fairness_loss(hidden_states, language_ids):
    """
    Fairness loss = variance of the average sentence embeddings across languages,
    using population variance (unbiased=False) and skipping if only one group.
    """
    sentence_embeddings = hidden_states.mean(dim=1)

    unique_langs = torch.unique(language_ids)
    if unique_langs.numel() < 2:
        return torch.tensor(0.0, device=hidden_states.device)

    group_means = []
    for lang in unique_langs:
        mask = (language_ids == lang)
        if mask.sum() > 0:
            group_means.append(sentence_embeddings[mask].mean(dim=0))

    if len(group_means) < 2:
        return torch.tensor(0.0, device=hidden_states.device)

    group_means = torch.stack(group_means)  

    var_loss = group_means.var(dim=0, unbiased=False).mean()
    return var_loss


def train_model(model, dataloader, optimizer, device, num_epochs=3, fairness_weight=0.2):
    """
    Training loop for the multilingual model.
    
    model: The masked language model.
    dataloader: DataLoader providing batches from MultilingualDataset.
    optimizer: Optimizer (e.g., AdamW) for model parameters.
    fairness_weight: Weight for the fairness loss component.
    """
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

            print(f"Epoch {epoch+1}: MLM Loss = {mlm_loss.item():.4f}, Fairness Loss = {fairness_loss.item():.4f}, Total Loss = {total_loss.item():.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "xlm-roberta-base"
    
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    model = XLMRobertaForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
    model.to(device)

    texts = [
        "Hello, this is an English sentence.",
        "Hola, esta es una oración en español.",
        "Bonjour, ceci est une phrase en français.",
        "你好，这是一个中文句子。",
        "नमस्ते, यह एक हिंदी वाक्य है।",
        "Hii ni sentensi ya Kiswahili"
    ]
 
    language_labels = [0, 1, 2, 3, 4, 5]

    augmented_texts = []
    augmented_labels = []
    for text, lang in zip(texts, language_labels):
        augmented_texts.append(text)
        augmented_labels.append(lang)
        if lang != 0 and random.random() > 0.5:
            augmented_version = augment_text(text)
            augmented_texts.append(augmented_version)
            augmented_labels.append(lang)

    dataset = MultilingualDataset(augmented_texts, augmented_labels, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    train_model(model, dataloader, optimizer, device, num_epochs=3, fairness_weight=0.1)
