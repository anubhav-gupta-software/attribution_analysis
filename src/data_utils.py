import os
import json
import urllib.request
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

def load_and_process_hatexplain():
    """
    Loads the hatexplain dataset from github and processes it into a pandas DataFrame.
    Returns the processed DataFrame.
    """
    print("Downloading HateXplain dataset.json...")
    url = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json"
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    
    all_data = []
    
    for post_id, item in data.items():
        post_tokens = item['post_tokens']
        text = " ".join(post_tokens)
        
        # Labels: 'hate speech', 'normal', 'offensive'
        labels = [ann['label'] for ann in item['annotators']]
        
        try:
            majority_label = Counter(labels).most_common(1)[0][0]
        except IndexError:
            continue
            
        # Map to binary
        binary_label = 1 if majority_label in ['hate speech', 'offensive'] else 0
        
        rationales = item.get('rationales', [])
        
        token_votes = np.zeros(len(post_tokens))
        num_annotators = len(rationales)
        
        for annotator_rat in rationales:
            for token_idx in annotator_rat:
                if int(token_idx) < len(post_tokens):
                    token_votes[int(token_idx)] += 1
                    
        if num_annotators > 0:
            threshold = num_annotators / 2.0
            majority_rationale = [1 if v > threshold else 0 for v in token_votes]
        else:
            majority_rationale = [0] * len(post_tokens)
            
        all_data.append({
            'post_id': post_id,
            'text': text,
            'post_tokens': post_tokens,
            'label': binary_label,
            'rationale': majority_rationale
        })
            
    df = pd.DataFrame(all_data)
    print(f"Loaded {len(df)} total examples.")
    return df

def create_splits(df, save_dir='data/processed'):
    os.makedirs(save_dir, exist_ok=True)
    
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    
    train_df.to_pickle(os.path.join(save_dir, 'train.pkl'))
    val_df.to_pickle(os.path.join(save_dir, 'val.pkl'))
    test_df.to_pickle(os.path.join(save_dir, 'test.pkl'))
    
    print(f"Saved splits to {save_dir}/")
    return train_df, val_df, test_df

if __name__ == "__main__":
    df = load_and_process_hatexplain()
    create_splits(df)
