import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
import evaluate
import numpy as np

class AlbertModel:
    def __init__(self, model_name='albert-base-v2', model_dir='models/albert'):
        self.model_name = model_name
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        os.makedirs(model_dir, exist_ok=True)
        
    def _compute_metrics(self, eval_preds):
        metric = evaluate.load("f1")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
        
    def train(self, df_train, df_val, epochs=3, batch_size=16):
        print(f"Loading {self.model_name}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=2
        )
        
        # Convert pandas to HF Dataset
        train_dataset = Dataset.from_pandas(df_train[['text', 'label']])
        val_dataset = Dataset.from_pandas(df_val[['text', 'label']])
        
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], truncation=True, max_length=128)
            
        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_val = val_dataset.map(tokenize_function, batched=True)
        
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        training_args = TrainingArguments(
            output_dir=self.model_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir='./logs_albert',
            logging_steps=50,
            seed=42,
            report_to="none" 
        )
        
        trainer_kwargs = dict(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
        )
        # transformers API changed: `tokenizer` -> `processing_class`.
        try:
            trainer = Trainer(processing_class=self.tokenizer, **trainer_kwargs)
        except TypeError:
            trainer = Trainer(tokenizer=self.tokenizer, **trainer_kwargs)
        
        print("Starting training...")
        trainer.train()
        
        print(f"Saving best model to {self.model_dir}...")
        trainer.save_model(self.model_dir)
        print("Training complete.")
        
    def load(self, path=None):
        load_path = path if path else self.model_dir
        self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        
    def predict(self, texts, batch_size=16):
        if self.model is None:
            raise ValueError("Model not loaded/trained")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()
        
        all_preds = []
        all_probs = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                
        return np.array(all_preds), np.array(all_probs)
