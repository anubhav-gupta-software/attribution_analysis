import json
import os

def create_nb(cells):
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    for cell_type, source in cells:
        nb["cells"].append({
            "cell_type": cell_type,
            "metadata": {},
            "execution_count": None if cell_type == "code" else None,
            "outputs": [] if cell_type == "code" else None,
            "source": [line + "\n" if not line.endswith("\n") else line for line in source.split("\n")]
        })
        if cell_type == "markdown":
            nb["cells"][-1].pop("execution_count")
            nb["cells"][-1].pop("outputs")
            
    return nb

roberta_cells = [
    ("markdown", "# RoBERTa Fine-tuning\nTraining the transformer model fully."),
    ("code", "import pandas as pd\nimport sys\nimport os\nsys.path.append(os.path.abspath('../src'))\nfrom models.roberta import RobertaModel"),
    ("code", "train_df = pd.read_pickle('../data/processed/train.pkl')\nval_df = pd.read_pickle('../data/processed/val.pkl')\n"),
    ("code", "model = RobertaModel(model_dir='../models/roberta')\nmodel.train(train_df, val_df, epochs=3, batch_size=16)"),
    ("code", "# For evaluation, load best model\ntest_df = pd.read_pickle('../data/processed/test.pkl')\nmodel.load()\npreds, probs = model.predict(test_df['text'].tolist(), batch_size=16)\nfrom evaluation import compute_classification_metrics\nprint('RoBERTa Test Metrics:', compute_classification_metrics(test_df['label'], preds))")
]

adv_cells = [
    ("markdown", "# Adversarial Evaluation\nApplying perturbations and checking Attack Success Rate (ASR) for both models."),
    ("code", "import pandas as pd\nimport sys\nimport os\nsys.path.append(os.path.abspath('../src'))\nfrom perturbations import generate_adversarial_dataset\nfrom models.lr_tfidf import LRTfidfModel\nfrom models.roberta import RobertaModel\nfrom evaluation import compute_attack_success_rate, compute_classification_metrics"),
    ("code", "adv_df = pd.read_pickle('../data/processed/adv_test.pkl')\norig_true = adv_df['label'].tolist()"),
    ("code", "lr_model = LRTfidfModel(model_dir='../models/lr')\nlr_model.load()\nlr_orig_preds = lr_model.predict(adv_df['original_text'].tolist())\nlr_adv_preds = lr_model.predict(adv_df['perturbed_text'].tolist())\n\nlr_asr = compute_attack_success_rate(orig_true, lr_orig_preds, lr_adv_preds)\nprint(f'Overall Attack Success Rate (LR Baseline): {lr_asr:.2%}')"),
    ("code", "rob_model = RobertaModel(model_dir='../models/roberta')\nrob_model.load()\nrob_orig_preds, _ = rob_model.predict(adv_df['original_text'].tolist(), batch_size=16)\nrob_adv_preds, _ = rob_model.predict(adv_df['perturbed_text'].tolist(), batch_size=16)\n\nrob_asr = compute_attack_success_rate(orig_true, rob_orig_preds, rob_adv_preds)\nprint(f'Overall Attack Success Rate (RoBERTa): {rob_asr:.2%}')")
]

attr_cells = [
    ("markdown", "# Attribution Alignment\nCheck how Integrated Gradients align with human rationales for Baseline and RoBERTa."),
    ("code", "import pandas as pd\nimport sys\nimport os\nimport numpy as np\nimport matplotlib.pyplot as plt\nsys.path.append(os.path.abspath('../src'))\nfrom attribution import LRAttribution, RobertaAttribution\nfrom evaluation import compute_attribution_iou, compute_attribution_shift"),
    ("code", "adv_df = pd.read_pickle('../data/processed/adv_test.pkl')\nrow = adv_df.iloc[0]\noriginal_text = row['original_text']\nperturbed_text = row['perturbed_text']"),
    ("code", "# Baseline Attribution\nlr_attr = LRAttribution(model_dir='../models/lr')\nlr_orig_tokens, lr_orig_scores = lr_attr.get_attribution(original_text)\nlr_adv_tokens, lr_adv_scores = lr_attr.get_attribution(perturbed_text)\nprint('--- LOGISTIC REGRESSION ATTRIBUTION ---')\nprint('Original tokens:', lr_orig_tokens)\nprint('Original attributions:', lr_orig_scores.round(2))\nprint('\\nAdversarial tokens:', lr_adv_tokens)\nprint('Adversarial attributions:', lr_adv_scores.round(2))\n"),
    ("code", "# RoBERTa Attribution\nrob_attr = RobertaAttribution(model_dir='../models/roberta')\nrob_orig_tokens, rob_orig_scores = rob_attr.get_attribution(original_text)\nrob_adv_tokens, rob_adv_scores = rob_attr.get_attribution(perturbed_text)\nprint('--- RoBERTa INTEGRATED GRADIENTS ---')\nprint('Original tokens:', rob_orig_tokens)\nprint('Original attributions:', rob_orig_scores.round(2))\nprint('\\nAdversarial tokens:', rob_adv_tokens)\nprint('Adversarial attributions:', rob_adv_scores.round(2))")
]

nbs = {
    "03_roberta.ipynb": roberta_cells,
    "04_adversarial.ipynb": adv_cells,
    "05_attribution.ipynb": attr_cells
}

for name, cells in nbs.items():
    nb_dict = create_nb(cells)
    with open(f"notebooks/{name}", "w") as f:
        json.dump(nb_dict, f, indent=2)

print("Notebooks updated for full execution!")
