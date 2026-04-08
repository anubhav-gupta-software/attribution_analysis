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

eda_cells = [
    ("markdown", "# Exploratory Data Analysis\nLet's load the data from `data/processed/` and look at class distribution and rationale lengths."),
    ("code", "import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport sys\nsys.path.append('../src')\nfrom data_utils import *"),
    ("code", "train_df = pd.read_pickle('../data/processed/train.pkl')\nval_df = pd.read_pickle('../data/processed/val.pkl')\ntest_df = pd.read_pickle('../data/processed/test.pkl')\nprint(f'Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}')"),
    ("code", "plt.figure(figsize=(6,4))\nsns.countplot(data=train_df, x='label')\nplt.title('Training Label Distribution (0: Non-Toxic, 1: Toxic)')\nplt.show()")
]

baseline_cells = [
    ("markdown", "# Baseline LR Model\nTraining the character n-gram TF-IDF Logistic Regression model."),
    ("code", "import pandas as pd\nimport sys\nimport os\nsys.path.append(os.path.abspath('../src'))\nfrom models.lr_tfidf import LRTfidfModel\nfrom evaluation import compute_classification_metrics"),
    ("code", "train_df = pd.read_pickle('../data/processed/train.pkl')\nval_df = pd.read_pickle('../data/processed/val.pkl')"),
    ("code", "model = LRTfidfModel(model_dir='../models/lr')\nmodel.train(train_df, val_df)"),
    ("code", "test_df = pd.read_pickle('../data/processed/test.pkl')\npreds = model.predict(test_df['text'])\nmetrics = compute_classification_metrics(test_df['label'], preds)\nprint('Baseline Test Metrics:', metrics)")
]

roberta_cells = [
    ("markdown", "# RoBERTa Fine-tuning\nTraining the transformer model."),
    ("code", "import pandas as pd\nimport sys\nimport os\nsys.path.append(os.path.abspath('../src'))\nfrom models.roberta import RobertaModel"),
    ("code", "train_df = pd.read_pickle('../data/processed/train.pkl')\nval_df = pd.read_pickle('../data/processed/val.pkl')\n\n# We take a small sample to show it works quickly. Remove .head() for real training\ntrain_sample = train_df.head(500)\nval_sample = val_df.head(100)"),
    ("code", "model = RobertaModel(model_dir='../models/roberta')\nmodel.train(train_sample, val_sample, epochs=1, batch_size=16)"),
    ("code", "# For evaluation, load best model\ntest_df = pd.read_pickle('../data/processed/test.pkl')\nmodel.load()\npreds, probs = model.predict(test_df.head(100)['text'].tolist())\nfrom evaluation import compute_classification_metrics\nprint('RoBERTa Validation Sample Metrics:', compute_classification_metrics(test_df.head(100)['label'], preds))")
]

adv_cells = [
    ("markdown", "# Adversarial Evaluation\nApplying perturbations and checking Attack Success Rate (ASR)."),
    ("code", "import pandas as pd\nimport sys\nimport os\nsys.path.append(os.path.abspath('../src'))\nfrom perturbations import generate_adversarial_dataset\nfrom models.lr_tfidf import LRTfidfModel\nfrom evaluation import compute_attack_success_rate, compute_classification_metrics"),
    ("code", "test_df = pd.read_pickle('../data/processed/test.pkl')\nadv_df = generate_adversarial_dataset(test_df, severity=0.3)\nadv_df.to_pickle('../data/processed/adv_test.pkl')\nadv_df.head()"),
    ("code", "model = LRTfidfModel(model_dir='../models/lr')\nmodel.load()\norig_preds = model.predict(adv_df['original_text'].tolist())\nadv_preds = model.predict(adv_df['perturbed_text'].tolist())\norig_true = adv_df['label'].tolist()"),
    ("code", "asr = compute_attack_success_rate(orig_true, orig_preds, adv_preds)\nprint(f'Overall Attack Success Rate (LR Baseline): {asr:.2%}')")
]

attr_cells = [
    ("markdown", "# Attribution Alignment\nCheck how Integrated Gradients align with human rationales."),
    ("code", "import pandas as pd\nimport sys\nimport os\nimport numpy as np\nimport matplotlib.pyplot as plt\nsys.path.append(os.path.abspath('../src'))\nfrom attribution import LRAttribution\nfrom evaluation import compute_attribution_iou, compute_attribution_shift"),
    ("code", "model_attr = LRAttribution(model_dir='../models/lr')\nadv_df = pd.read_pickle('../data/processed/adv_test.pkl')\nrow = adv_df.iloc[0]\noriginal_text = row['original_text']\nperturbed_text = row['perturbed_text']"),
    ("code", "orig_tokens, orig_scores = model_attr.get_attribution(original_text)\nadv_tokens, adv_scores = model_attr.get_attribution(perturbed_text)\nprint('Original tokens:', orig_tokens)\nprint('Original attributions:', orig_scores.round(2))\n\nprint('\\nAdversarial tokens:', adv_tokens)\nprint('Adversarial attributions:', adv_scores.round(2))")
]

nbs = {
    "01_EDA.ipynb": eda_cells,
    "02_baseline.ipynb": baseline_cells,
    "03_roberta.ipynb": roberta_cells,
    "04_adversarial.ipynb": adv_cells,
    "05_attribution.ipynb": attr_cells
}

for name, cells in nbs.items():
    nb_dict = create_nb(cells)
    with open(f"notebooks/{name}", "w") as f:
        json.dump(nb_dict, f, indent=2)

print("Notebooks populated!")
