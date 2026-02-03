"""
Advanced Medical NLP - Model Fine-tuning and Training
This module provides utilities for fine-tuning models on medical data
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import json


class MedicalSentimentDataset(Dataset):
    """Custom dataset for medical sentiment analysis"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class MedicalNERDataset(Dataset):
    """Custom dataset for medical Named Entity Recognition"""
    
    def __init__(self, texts: List[List[str]], tags: List[List[int]], tokenizer, max_length: int = 128):
        self.texts = texts
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        words = self.texts[idx]
        tags = self.tags[idx]
        
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Align tags with tokenized input
        labels = self._align_labels(encoding, tags)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
    def _align_labels(self, encoding, tags):
        """Align labels with tokenized input"""
        word_ids = encoding.word_ids()
        labels = []
        
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)  # Ignore padding tokens
            else:
                labels.append(tags[word_id])
        
        return labels


class MedicalNLPTrainer:
    """Trainer class for medical NLP models"""
    
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def prepare_sentiment_data(self, data_file: str = None) -> Tuple[Dataset, Dataset]:
        """Prepare sentiment analysis dataset"""
        
        # Example synthetic data (in practice, load from actual medical dataset)
        sample_data = [
            {"text": "I'm very worried about my recovery after surgery", "label": 0},  # Anxious
            {"text": "The doctor reassured me everything is fine", "label": 2},  # Reassured
            {"text": "I have some pain but it's manageable", "label": 1},  # Neutral
            {"text": "I'm scared about the test results", "label": 0},  # Anxious
            {"text": "Feeling much better after the treatment", "label": 2},  # Reassured
            {"text": "The medication is working well", "label": 2},  # Reassured
            {"text": "I'm not sure if this is normal", "label": 1},  # Neutral
            {"text": "Very concerned about side effects", "label": 0},  # Anxious
        ] * 100  # Multiply for training purposes
        
        texts = [item["text"] for item in sample_data]
        labels = [item["label"] for item in sample_data]
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # Create datasets
        train_dataset = MedicalSentimentDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = MedicalSentimentDataset(val_texts, val_labels, self.tokenizer)
        
        return train_dataset, val_dataset
    
    def train_sentiment_model(self, output_dir: str = "./medical_sentiment_model"):
        """Fine-tune model for medical sentiment analysis"""
        
        print("Preparing sentiment analysis data...")
        train_dataset, val_dataset = self.prepare_sentiment_data()
        
        print("Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3,  # Anxious, Neutral, Reassured
            id2label={0: "Anxious", 1: "Neutral", 2: "Reassured"},
            label2id={"Anxious": 0, "Neutral": 1, "Reassured": 2}
        )
        
        print("Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy"
        )
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            accuracy = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average='weighted')
            
            return {
                'accuracy': accuracy,
                'f1': f1
            }
        
        print("Initializing trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer)
        )
        
        print("Starting training...")
        trainer.train()
        
        print(f"Model saved to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Evaluate
        print("\nEvaluating model...")
        eval_results = trainer.evaluate()
        print(f"Evaluation results: {eval_results}")
        
        return trainer, eval_results
    
    def prepare_ner_data(self) -> Tuple[Dataset, Dataset]:
        """Prepare NER dataset for medical entities"""
        
        # Example NER data
        # Tags: 0=O, 1=B-SYMPTOM, 2=I-SYMPTOM, 3=B-DIAGNOSIS, 4=I-DIAGNOSIS, 5=B-TREATMENT, 6=I-TREATMENT
        
        sample_data = [
            {
                "tokens": ["Patient", "has", "severe", "neck", "pain", "and", "headache"],
                "tags": [0, 0, 1, 2, 2, 0, 1]
            },
            {
                "tokens": ["Diagnosed", "with", "whiplash", "injury"],
                "tags": [0, 0, 3, 4]
            },
            {
                "tokens": ["Treatment", "includes", "physiotherapy", "sessions"],
                "tags": [0, 0, 5, 6]
            }
        ] * 100
        
        texts = [item["tokens"] for item in sample_data]
        tags = [item["tags"] for item in sample_data]
        
        # Split data
        train_texts, val_texts, train_tags, val_tags = train_test_split(
            texts, tags, test_size=0.2, random_state=42
        )
        
        # Create datasets
        train_dataset = MedicalNERDataset(train_texts, train_tags, self.tokenizer)
        val_dataset = MedicalNERDataset(val_texts, val_tags, self.tokenizer)
        
        return train_dataset, val_dataset
    
    def train_ner_model(self, output_dir: str = "./medical_ner_model"):
        """Fine-tune model for medical NER"""
        
        print("Preparing NER data...")
        train_dataset, val_dataset = self.prepare_ner_data()
        
        # Define label list
        label_list = ["O", "B-SYMPTOM", "I-SYMPTOM", "B-DIAGNOSIS", "I-DIAGNOSIS", "B-TREATMENT", "I-TREATMENT"]
        
        print("Loading model...")
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(label_list),
            id2label={i: label for i, label in enumerate(label_list)},
            label2id={label: i for i, label in enumerate(label_list)}
        )
        
        print("Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True
        )
        
        print("Initializing trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer)
        )
        
        print("Starting training...")
        trainer.train()
        
        print(f"Model saved to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        return trainer


class DatasetCreator:
    """Utilities for creating medical NLP datasets"""
    
    @staticmethod
    def create_sentiment_dataset_template() -> pd.DataFrame:
        """Create template for sentiment annotation"""
        template = pd.DataFrame({
            'text': [
                'Example patient dialogue 1',
                'Example patient dialogue 2',
                'Example patient dialogue 3'
            ],
            'sentiment': ['anxious', 'neutral', 'reassured'],
            'intent': ['seeking_reassurance', 'reporting_symptoms', 'asking_questions'],
            'confidence': [0.9, 0.8, 0.95]
        })
        return template
    
    @staticmethod
    def create_ner_dataset_template() -> List[Dict]:
        """Create template for NER annotation"""
        template = [
            {
                'text': 'Patient has severe neck pain',
                'entities': [
                    {'start': 12, 'end': 18, 'label': 'SYMPTOM', 'text': 'severe'},
                    {'start': 19, 'end': 28, 'label': 'SYMPTOM', 'text': 'neck pain'}
                ]
            }
        ]
        return template
    
    @staticmethod
    def create_soap_dataset_template() -> Dict:
        """Create template for SOAP note annotation"""
        template = {
            'transcript': 'Doctor: ... Patient: ...',
            'soap_note': {
                'subjective': 'Patient complains of...',
                'objective': 'Physical examination reveals...',
                'assessment': 'Diagnosis: ...',
                'plan': 'Treatment plan: ...'
            }
        }
        return template
    
    @staticmethod
    def save_templates(output_dir: str = "./dataset_templates"):
        """Save all dataset templates"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Sentiment template
        sentiment_df = DatasetCreator.create_sentiment_dataset_template()
        sentiment_df.to_csv(f"{output_dir}/sentiment_template.csv", index=False)
        
        # NER template
        ner_template = DatasetCreator.create_ner_dataset_template()
        with open(f"{output_dir}/ner_template.json", 'w') as f:
            json.dump(ner_template, f, indent=2)
        
        # SOAP template
        soap_template = DatasetCreator.create_soap_dataset_template()
        with open(f"{output_dir}/soap_template.json", 'w') as f:
            json.dump(soap_template, f, indent=2)
        
        print(f"Templates saved to {output_dir}/")


def main():
    """Main function for training models"""
    
    print("Medical NLP Model Training")
    print("=" * 80)
    
    # Initialize trainer
    trainer = MedicalNLPTrainer()
    
    # Option 1: Train sentiment model
    print("\n1. Training Sentiment Analysis Model...")
    print("-" * 80)
    sentiment_trainer, results = trainer.train_sentiment_model()
    
    # Option 2: Train NER model (commented out by default)
    # print("\n2. Training NER Model...")
    # print("-" * 80)
    # ner_trainer = trainer.train_ner_model()
    
    # Create dataset templates
    print("\n3. Creating Dataset Templates...")
    print("-" * 80)
    DatasetCreator.save_templates()
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
