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

tokenizers_cells = [
    ("code", "!pip install transformers datasets evaluate captum scikit-learn pandas -q"),
    ("markdown", "# Sub-Word Tokenization Fine-tuning\nTraining all transformer models fully: BPE (RoBERTa), WordPiece (BERT), and SentencePiece (ALBERT)."),
    ("code", "import pandas as pd\nimport sys\nimport os\nsys.path.append(os.path.abspath('../src'))\nfrom models.roberta import RobertaModel\nfrom models.bert import BertModel\nfrom models.albert import AlbertModel"),
    ("code", "train_df = pd.read_pickle('../data/processed/train.pkl')\nval_df = pd.read_pickle('../data/processed/val.pkl')\n"),
    ("code", "# Train RoBERTa\nroberta_model = RobertaModel(model_dir='../models/roberta')\nroberta_model.train(train_df, val_df, epochs=3, batch_size=16)"),
    ("code", "# Train BERT\nbert_model = BertModel(model_dir='../models/bert')\nbert_model.train(train_df, val_df, epochs=3, batch_size=16)"),
    ("code", "# Train ALBERT\nalbert_model = AlbertModel(model_dir='../models/albert')\nalbert_model.train(train_df, val_df, epochs=4, batch_size=16)")
]

adv_cells = [
    ("markdown", "# Google Colab Setup\nRun this cell to install all required dependencies."),
    ("code", "!pip install transformers datasets evaluate captum scikit-learn pandas -q"),
    ("markdown", "# Adversarial Evaluation\nApplying perturbations and checking Attack Success Rate (ASR) across all 5 architectures."),
    ("code", "import pandas as pd\nimport sys\nimport os\nsys.path.append(os.path.abspath('../src'))\nfrom models.lr_tfidf import LRTfidfModel\nfrom models.roberta import RobertaModel\nfrom models.bert import BertModel\nfrom models.albert import AlbertModel\nfrom evaluation import compute_attack_success_rate"),
    ("code", "adv_df = pd.read_pickle('../data/processed/adv_test.pkl')\norig_true = adv_df['label'].tolist()"),
    ("code", "models = {\n    'lr_char': LRTfidfModel(model_dir='../models/lr_char', analyzer='char_wb'),\n    'lr_word': LRTfidfModel(model_dir='../models/lr_word', analyzer='word'),\n    'roberta': RobertaModel(model_dir='../models/roberta'),\n    'bert': BertModel(model_dir='../models/bert'),\n    'albert': AlbertModel(model_dir='../models/albert')\n}"),
    ("code", "for name, model in models.items():\n    model.load()\n    if 'lr' in name:\n        orig_preds = model.predict(adv_df['original_text'].tolist())\n        adv_preds = model.predict(adv_df['perturbed_text'].tolist())\n    else:\n        orig_preds, _ = model.predict(adv_df['original_text'].tolist(), batch_size=16)\n        adv_preds, _ = model.predict(adv_df['perturbed_text'].tolist(), batch_size=16)\n    asr = compute_attack_success_rate(orig_true, orig_preds, adv_preds)\n    print(f'Overall Attack Success Rate ({name.upper()}): {asr:.2%}')")
]

attr_cells = [
    ("markdown", "# Google Colab Setup\nRun this cell to install all required dependencies."),
    ("code", "!pip install transformers datasets evaluate captum scikit-learn pandas -q"),
    ("markdown", "# Attribution Alignment\nCheck how Integrated Gradients align with human rationales across all 5 Tokenizers."),
    ("code", "import pandas as pd\nimport sys\nimport os\nsys.path.append(os.path.abspath('../src'))\nfrom attribution import LRAttribution, RobertaAttribution, BertAttribution, AlbertAttribution"),
    ("code", "adv_df = pd.read_pickle('../data/processed/adv_test.pkl')\nrow = adv_df.iloc[0]\noriginal_text = row['original_text']\nperturbed_text = row['perturbed_text']"),
    ("code", "attributions = {\n    'lr_char': LRAttribution(model_dir='../models/lr_char'),\n    'lr_word': LRAttribution(model_dir='../models/lr_word'),\n    'roberta': RobertaAttribution(model_dir='../models/roberta'),\n    'bert': BertAttribution(model_dir='../models/bert'),\n    'albert': AlbertAttribution(model_dir='../models/albert')\n}"),
    ("code", "for name, attr_model in attributions.items():\n    orig_tokens, orig_scores = attr_model.get_attribution(original_text)\n    adv_tokens, adv_scores = attr_model.get_attribution(perturbed_text)\n    print(f'\\n--- {name.upper()} INTEGRATED GRADIENTS ---')\n    print('Original tokens:', orig_tokens)\n    print('Adversarial tokens:', adv_tokens)")
]

nbs = {
    "03_transformers.ipynb": tokenizers_cells,
    "04_adversarial.ipynb": adv_cells,
    "05_attribution.ipynb": attr_cells
}

for name, cells in nbs.items():
    nb_dict = create_nb(cells)
    with open(f"notebooks/{name}", "w") as f:
        json.dump(nb_dict, f, indent=2)

print("Notebooks updated for full 5-model execution!")
