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
    ("markdown", "# Sub-Word Tokenization Fine-tuning\nTraining all transformer models: BPE (RoBERTa), WordPiece (BERT), and SentencePiece (ALBERT)."),
    ("code", "import pandas as pd\nimport sys\nimport os\nsys.path.append(os.path.abspath('../src'))\nfrom models.roberta import RobertaModel\nfrom models.bert import BertModel\nfrom models.albert import AlbertModel"),
    ("code", "train_df = pd.read_pickle('../data/processed/train.pkl')\nval_df = pd.read_pickle('../data/processed/val.pkl')\n"),
    ("code", "# Train RoBERTa\nroberta_model = RobertaModel(model_dir='../models/roberta')\nroberta_model.train(train_df, val_df, epochs=3, batch_size=16)"),
    ("code", "# Train BERT\nbert_model = BertModel(model_dir='../models/bert')\nbert_model.train(train_df, val_df, epochs=3, batch_size=16)"),
    ("code", "# Train ALBERT\nalbert_model = AlbertModel(model_dir='../models/albert')\nalbert_model.train(train_df, val_df, epochs=4, batch_size=16)")
]

adv_cells = [
    ("code", "!pip install transformers datasets evaluate captum scikit-learn pandas -q"),
    ("markdown", "# Adversarial Evaluation\nApplying perturbations and checking Attack Success Rate (ASR) across all 5 architectures."),
    ("code", "import pandas as pd\nimport sys\nimport os\nsys.path.append(os.path.abspath('../src'))\nfrom models.lr_tfidf import LRTfidfModel\nfrom models.roberta import RobertaModel\nfrom models.bert import BertModel\nfrom models.albert import AlbertModel\nfrom evaluation import compute_attack_success_rate"),
    ("code", "adv_df = pd.read_pickle('../data/processed/adv_test.pkl')\norig_true = adv_df['label'].tolist()"),
    ("code", "models = {\n    'lr_char': LRTfidfModel(model_dir='../models/lr_char', analyzer='char_wb'),\n    'lr_word': LRTfidfModel(model_dir='../models/lr_word', analyzer='word'),\n    'roberta': RobertaModel(model_dir='../models/roberta'),\n    'bert': BertModel(model_dir='../models/bert'),\n    'albert': AlbertModel(model_dir='../models/albert')\n}"),
    ("code", "for name, model in models.items():\n    model.load()\n    if 'lr' in name:\n        orig_preds = model.predict(adv_df['original_text'].tolist())\n        adv_preds = model.predict(adv_df['perturbed_text'].tolist())\n    else:\n        orig_preds, _ = model.predict(adv_df['original_text'].tolist(), batch_size=16)\n        adv_preds, _ = model.predict(adv_df['perturbed_text'].tolist(), batch_size=16)\n    asr = compute_attack_success_rate(orig_true, orig_preds, adv_preds)\n    print(f'Overall Attack Success Rate ({name.upper()}): {asr:.2%}')")
]

attr_cells = [
    ("code", "!pip install transformers datasets evaluate captum scikit-learn pandas -q"),
    ("markdown", "# Attribution Analysis\nIG attributions vs human rationales across all tokenizers."),
    ("code", "import pandas as pd\nimport numpy as np\nimport time\nimport sys\nimport os\nsys.path.append(os.path.abspath('../src'))\nfrom attribution import LRAttribution, RobertaAttribution, BertAttribution, AlbertAttribution\nfrom evaluation import aggregate_word_attributions, compute_token_f1\nfrom transformers import AutoTokenizer"),
    ("code", "adv_df = pd.read_pickle('../data/processed/adv_test.pkl')\nrow = adv_df.iloc[0]\noriginal_text = row['original_text']\nperturbed_text = row['perturbed_text']"),
    ("code", "attributions = {\n    'lr_char': LRAttribution(model_dir='../models/lr_char'),\n    'lr_word': LRAttribution(model_dir='../models/lr_word'),\n    'roberta': RobertaAttribution(model_dir='../models/roberta'),\n    'bert': BertAttribution(model_dir='../models/bert'),\n    'albert': AlbertAttribution(model_dir='../models/albert')\n}"),
    ("code", "for name, attr_model in attributions.items():\n    orig_tokens, orig_scores = attr_model.get_attribution(original_text)\n    adv_tokens, adv_scores = attr_model.get_attribution(perturbed_text)\n    print(f'\\n--- {name.upper()} ---')\n    print('Original:', orig_tokens)\n    print('Adversarial:', adv_tokens)"),
    ("markdown", "# Token-Level F1\nCompare IG attribution alignment with human rationales before and after adversarial attack.\nspace_injection is excluded from Token-F1 because it breaks word-level rationale alignment."),
    ("code", """test_df = pd.read_pickle('../data/processed/test.pkl')
toxic_test = test_df[(test_df['label'] == 1) & (test_df['rationale'].apply(lambda r: sum(r) > 0))].copy()
print(f'Toxic samples with rationales: {len(toxic_test)}')"""),
    ("code", """model_configs = {
    'RoBERTa': (RobertaAttribution, '../models/roberta'),
    'BERT': (BertAttribution, '../models/bert'),
    'ALBERT': (AlbertAttribution, '../models/albert'),
}

all_results = {}"""),
    ("code", """for model_name, (attr_class, model_dir) in model_configs.items():
    print(f'\\n{"="*60}')
    print(f'{model_name}')
    print(f'{"="*60}')
    
    attr_model = attr_class(model_dir=model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    orig_results = []
    adv_results = {a: [] for a in ['char_insertion', 'leetspeak', 'homoglyph_swap']}
    
    start = time.time()
    
    for idx, (_, row) in enumerate(toxic_test.iterrows()):
        if idx % 50 == 0:
            print(f'  {idx}/{len(toxic_test)} ({time.time()-start:.0f}s)')
        
        try:
            _, scores = attr_model.get_attribution(row['text'], target_class=1)
            word_attrs = aggregate_word_attributions(tokenizer, row['text'], scores)
            r = compute_token_f1(word_attrs, row['rationale'])
            if r: orig_results.append(r)
        except:
            continue
        
        for _, adv_row in adv_df[adv_df['post_id'] == row['post_id']].iterrows():
            if adv_row['attack_type'] == 'space_injection':
                continue
            try:
                _, adv_scores = attr_model.get_attribution(adv_row['perturbed_text'], target_class=1)
                adv_word_attrs = aggregate_word_attributions(tokenizer, adv_row['perturbed_text'], adv_scores)
                r = compute_token_f1(adv_word_attrs, row['rationale'])
                if r: adv_results[adv_row['attack_type']].append(r)
            except:
                continue
    
    print(f'  Done: {len(orig_results)} orig + {sum(len(v) for v in adv_results.values())} adv ({time.time()-start:.0f}s)')
    all_results[model_name] = {'original': orig_results, 'adversarial': adv_results}"""),
    ("markdown", "## Results"),
    ("code", """for model_name in ['RoBERTa', 'BERT', 'ALBERT']:
    data = all_results[model_name]
    orig = data['original']
    
    orig_f1 = np.mean([r['f1'] for r in orig])
    orig_p = np.mean([r['precision'] for r in orig])
    orig_r = np.mean([r['recall'] for r in orig])
    
    all_adv = [r for v in data['adversarial'].values() for r in v]
    adv_f1 = np.mean([r['f1'] for r in all_adv])
    adv_p = np.mean([r['precision'] for r in all_adv])
    adv_r = np.mean([r['recall'] for r in all_adv])
    
    print(f'\\n{model_name}')
    print(f'  Original  - P: {orig_p:.3f}  R: {orig_r:.3f}  F1: {orig_f1:.3f}  (n={len(orig)})')
    print(f'  Adversarial - P: {adv_p:.3f}  R: {adv_r:.3f}  F1: {adv_f1:.3f}  (n={len(all_adv)})')
    print(f'  Delta F1: {adv_f1 - orig_f1:+.3f}')
    
    for attack in ['char_insertion', 'leetspeak', 'homoglyph_swap']:
        results = data['adversarial'][attack]
        if results:
            print(f'    {attack}: F1={np.mean([r[\"f1\"] for r in results]):.3f} (n={len(results)})')"""),
    ("code", """import json
save_data = {}
for m, data in all_results.items():
    save_data[m] = {'original': data['original'], 'adversarial': data['adversarial']}
with open('../results/token_f1_results.json', 'w') as f:
    json.dump(save_data, f, indent=2, default=str)
print('Saved to results/token_f1_results.json')""")
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


