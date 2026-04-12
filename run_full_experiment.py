import pandas as pd
import sys
import os
import time
sys.path.append(os.path.abspath('src'))
from models.roberta import RobertaModel
from models.lr_tfidf import LRTfidfModel
from evaluation import compute_classification_metrics, compute_attack_success_rate
from attribution import RobertaAttribution

print("=== 1. Starting Full RoBERTa Fine-tuning ===")
start_time = time.time()
train_df = pd.read_pickle('data/processed/train.pkl')
val_df = pd.read_pickle('data/processed/val.pkl')

# Train fully!
rob_model = RobertaModel(model_dir='models/roberta')
rob_model.train(train_df, val_df, epochs=3, batch_size=16)
print(f"Training Complete! Time elapsed: {(time.time()-start_time)/60:.2f} mins")

print("\n=== 2. Evaluating Adversarial Robustness ===")
test_df = pd.read_pickle('data/processed/test.pkl')
adv_df = pd.read_pickle('data/processed/adv_test.pkl')

rob_model.load()
print("Predicting against original test strings...")
rob_orig_preds, _ = rob_model.predict(adv_df['original_text'].tolist(), batch_size=32)

print("Predicting against perturbed strings...")
rob_adv_preds, _ = rob_model.predict(adv_df['perturbed_text'].tolist(), batch_size=32)

orig_true = adv_df['label'].tolist()
rob_asr = compute_attack_success_rate(orig_true, rob_orig_preds, rob_adv_preds)
print(f"\n[FINAL METRIC] Overall Attack Success Rate (RoBERTa): {rob_asr:.2%}")

print("\n=== 3. Extracting Attribution Shift ===")
rob_attr = RobertaAttribution(model_dir='models/roberta')
row = adv_df.iloc[0]
original_text = row['original_text']
perturbed_text = row['perturbed_text']

rob_orig_tokens, rob_orig_scores = rob_attr.get_attribution(original_text)
rob_adv_tokens, rob_adv_scores = rob_attr.get_attribution(perturbed_text)

print("\n--- RoBERTa INTEGRATED GRADIENTS ---")
print("Original tokens:", rob_orig_tokens)
print("Original attributions:", rob_orig_scores.round(2))
print("\nAdversarial tokens:", rob_adv_tokens)
print("Adversarial attributions:", rob_adv_scores.round(2))
print("\nFINISHED FULL RUN!")
