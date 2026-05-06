import random
import re
import pandas as pd

random.seed(42)

def char_insertion(word, char='.'):
    """Inserts a character between letters, keeping first and last letter intact."""
    if len(word) <= 2:
        return word
    return word[0] + char + char.join(list(word[1:-1])) + char + word[-1]

def space_injection(word):
    """Inserts space between characters."""
    if len(word) <= 2:
        return word
    return " ".join(list(word))

def leetspeak(word):
    """Replaces letters with similar looking numbers."""
    leet_map = {'a': '4', 'e': '3', 'i': '1', 'o': '0', 's': '5', 't': '7'}
    res = ""
    for c in word:
        res += leet_map.get(c.lower(), c)
    return res

def homoglyph_swap(word):
    """Replaces letters with similar looking symbols."""
    homo_map = {'a': '@', 'i': '!', 's': '$', 'l': '|', 'c': '(', 'o': '*'}
    res = ""
    for c in word:
        res += homo_map.get(c.lower(), c)
    return res

def generate_adversarial_dataset(df, severity=0.3):
    """
    Generates adversarial examples for the toxic comments in the dataframe.
    Severity specifies the fraction of alphabetic tokens to perturb.
    """
    toxic_df = df[df['label'] == 1].copy()
    
    results = []
    
    attacks = {
        'char_insertion': lambda w: char_insertion(w, '.'),
        'space_injection': space_injection,
        'leetspeak': leetspeak,
        'homoglyph_swap': homoglyph_swap
    }
    
    for i, row in toxic_df.iterrows():
        tokens = row['post_tokens']
        if not tokens:
            continue
            
        valid_indices = [idx for idx, t in enumerate(tokens) if re.match(r'^[a-zA-Z]+$', t) and len(t) > 2]
        if not valid_indices:
            continue
            
        num_perturb = max(1, int(len(valid_indices) * severity))
        num_perturb = min(num_perturb, len(valid_indices))
        
        for attack_name, attack_fn in attacks.items():
            perturbed_tokens = list(tokens)
            chosen_indices = random.sample(valid_indices, num_perturb)
            
            for idx in chosen_indices:
                perturbed_tokens[idx] = attack_fn(perturbed_tokens[idx])
                
            results.append({
                'post_id': row['post_id'],
                'original_text': row['text'],
                'perturbed_text': " ".join(perturbed_tokens), # Re-join for predictions
                # Keep original tokens for rationale alignment if needed, though they aren't strictly text aligned anymore
                'original_tokens': tokens,
                'perturbed_tokens': perturbed_tokens, 
                'label': 1,
                'attack_type': attack_name,
                'severity': severity
            })
            
    res_df = pd.DataFrame(results)
    print(f"Generated {len(res_df)} adversarial examples ({len(attacks)} attacks per toxic string).")
    return res_df

if __name__ == "__main__":
    import os
    if os.path.exists('data/processed/test.pkl'):
        test_df = pd.read_pickle('data/processed/test.pkl')
        adv_df = generate_adversarial_dataset(test_df, severity=0.3)
        adv_df.to_pickle('data/processed/adv_test.pkl')
        print("Saved generated adversarial dataset to data/processed/adv_test.pkl")
    else:
        print("Please run data_utils.py first to generate data/processed/test.pkl")
