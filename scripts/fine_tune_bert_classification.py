import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def fine_tune_bert(model_name='bert-base-uncased', num_labels=2, epochs=2, output_dir="./models/bert_classification"):
    # Vérifier GPU dispo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entraînement sur : {device}")

    # Tokenizer et modèle
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)

    # Charger le dataset IMDb
    dataset = load_dataset("imdb")

    # Tokenizer les textes
    tokenized_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Subset pour test rapide (optionnel)
    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(5000))
    test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    # Arguments d'entraînement
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir=os.path.join(output_dir, 'logs'),
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Entraînement
    trainer.train()

    # Sauvegarde du modèle
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Modèle et tokenizer sauvegardés dans : {output_dir}")

if __name__ == "__main__":
    fine_tune_bert()
