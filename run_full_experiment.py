import pandas as pd
import sys
import os
import time
sys.path.append(os.path.abspath('src'))
from models.roberta import RobertaModel
from models.bert import BertModel
from models.albert import AlbertModel
from models.lr_tfidf import LRTfidfModel
from evaluation import compute_classification_metrics, compute_attack_success_rate
from attribution import RobertaAttribution, BertAttribution, AlbertAttribution, LRAttribution

print("=== 1. Starting 5-Model Execution Pipeline ===")
train_df = pd.read_pickle('data/processed/train.pkl')
val_df = pd.read_pickle('data/processed/val.pkl')
test_df = pd.read_pickle('data/processed/test.pkl')
adv_df = pd.read_pickle('data/processed/adv_test.pkl')
orig_true = adv_df['label'].tolist()

models = {
    'lr_char': LRTfidfModel(model_dir='models/lr_char', analyzer='char_wb'),
    'lr_word': LRTfidfModel(model_dir='models/lr_word', analyzer='word'),
    'roberta': RobertaModel(model_dir='models/roberta'),
    'bert': BertModel(model_dir='models/bert'),
    'albert': AlbertModel(model_dir='models/albert')
}

asr_results = {}

for name, model in models.items():
    print(f"\n--- Training {name.upper()} ---")
    start_time = time.time()
    
    if 'lr' in name:
        model.train(train_df, val_df)
    else:
        model.train(train_df, val_df, epochs=3, batch_size=16)
        
    print(f"Training Complete for {name}! Time elapsed: {(time.time()-start_time)/60:.2f} mins")

    print(f"--- Evaluating {name.upper()} ---")
    model.load()
    if 'lr' in name:
        orig_preds = model.predict(adv_df['original_text'].tolist())
        adv_preds = model.predict(adv_df['perturbed_text'].tolist())
    else:
        orig_preds, _ = model.predict(adv_df['original_text'].tolist(), batch_size=32)
        adv_preds, _ = model.predict(adv_df['perturbed_text'].tolist(), batch_size=32)
        
    asr = compute_attack_success_rate(orig_true, orig_preds, adv_preds)
    asr_results[name] = asr
    print(f"[FINAL METRIC] Attack Success Rate ({name.upper()}): {asr:.2%}")

print("\n=== FINAL ASR COMPARISON ===")
for name, asr in asr_results.items():
    print(f"{name.upper()}: {asr:.2%}")

print("\n=== 2. Extracting Attribution Shifts (Demonstration) ===")
attributions = {
    'lr_char': LRAttribution(model_dir='models/lr_char'),
    'lr_word': LRAttribution(model_dir='models/lr_word'),
    'roberta': RobertaAttribution(model_dir='models/roberta'),
    'bert': BertAttribution(model_dir='models/bert'),
    'albert': AlbertAttribution(model_dir='models/albert')
}

row = adv_df.iloc[0]
original_text = row['original_text']
perturbed_text = row['perturbed_text']

for name, attr_model in attributions.items():
    orig_tokens, orig_scores = attr_model.get_attribution(original_text)
    adv_tokens, adv_scores = attr_model.get_attribution(perturbed_text)
    print(f"\n--- {name.upper()} INTEGRATED GRADIENTS ---")
    print("Original tokens:", orig_tokens)
    print("Original attributions:", orig_scores.round(2))
    print("\nAdversarial tokens:", adv_tokens)
    print("Adversarial attributions:", adv_scores.round(2))

print("\nFINISHED FULL 5-MODEL EXPERIMENT PIPELINE!")
