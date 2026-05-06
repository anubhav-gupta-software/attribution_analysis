import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_classification_metrics(y_true, y_pred):
    """
    Computes standard classification metrics.
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

def compute_attack_success_rate(y_true_orig, y_pred_orig, y_pred_adv):
    """
    Computes ASR (Attack Success Rate), which is the fraction of 
    examples originally classified correctly as toxic (1) that are now 
    misclassified as non-toxic (0).
    """
    successful_attacks = 0
    total_valid_examples = 0
    
    for t_orig, p_orig, p_adv in zip(y_true_orig, y_pred_orig, y_pred_adv):
        if t_orig == 1 and p_orig == 1:
            total_valid_examples += 1
            if p_adv == 0:
                successful_attacks += 1
                
    if total_valid_examples == 0:
        return 0.0
    return successful_attacks / total_valid_examples
    
def get_top_k_indices(attributions, k):
    """
    Returns the indices of the top k highest attribution scores.
    """
    return np.argsort(attributions)[-k:]

def compute_attribution_iou(attributions, rationale):
    """
    Computes Token IOU between binarized attributions and ground truth rationale.
    Binarizes attribution by taking top k tokens where k is the number of tokens 
    in the human rationale.
    """
    k = sum(rationale)
    if k == 0:
        return 0.0 # Cannot compute meaningfully if no rationale
        
    top_k_idx = get_top_k_indices(attributions, int(k))
    binarized_attr = np.zeros_like(attributions)
    binarized_attr[top_k_idx] = 1
    
    intersection = np.logical_and(binarized_attr, rationale).sum()
    union = np.logical_or(binarized_attr, rationale).sum()
    
    if union == 0:
        return 0.0
    return intersection / union

def compute_attribution_shift(attr_orig, attr_adv):
    """
    Computes cosine similarity between token attributions of original and adversarial text.
    Assumes attributions have been aligned (e.g. by padding or matching tokens).
    """
    norm_orig = np.linalg.norm(attr_orig)
    norm_adv = np.linalg.norm(attr_adv)
    if norm_orig == 0 or norm_adv == 0:
        return 0.0
    return np.dot(attr_orig, attr_adv) / (norm_orig * norm_adv)

def aggregate_word_attributions(tokenizer, text, token_attributions, max_length=128):
    """
    Maps sub-word token attributions back to word-level by summing absolute
    attribution values for all sub-tokens belonging to the same word.
    Uses HuggingFace tokenizer's word_ids() for alignment.
    
    Args:
        tokenizer: HuggingFace tokenizer instance
        text: the raw input string
        token_attributions: np.array of shape (seq_len,) from IG
        max_length: must match the max_length used during attribution
    
    Returns:
        word_attrs: np.array of shape (num_words,) with aggregated attributions
    """
    words = text.split()
    num_words = len(words)
    word_attrs = np.zeros(num_words)
    
    encoding = tokenizer(text, truncation=True, max_length=max_length)
    word_ids = encoding.word_ids()  # None for special tokens, int for word index
    
    for token_idx, word_id in enumerate(word_ids):
        if word_id is not None and word_id < num_words and token_idx < len(token_attributions):
            word_attrs[word_id] += abs(token_attributions[token_idx])
    
    return word_attrs

def compute_token_f1(word_attributions, rationale):
    """
    Computes token-level Precision, Recall, and F1 between binarized
    top-k attribution tokens and human rationale tokens.
    
    Binarization: selects top-k word positions by attribution magnitude,
    where k = number of rationale tokens (sum of rationale vector).
    
    Args:
        word_attributions: np.array of word-level attribution scores (num_words,)
        rationale: list/np.array of binary rationale labels (num_words,)
    
    Returns:
        dict with 'precision', 'recall', 'f1', or None if rationale is empty
    """
    rationale = np.array(rationale)
    k = int(rationale.sum())
    
    if k == 0 or len(word_attributions) == 0:
        return None
    
    # Handle length mismatch (truncation): trim to shorter length
    min_len = min(len(word_attributions), len(rationale))
    word_attributions = word_attributions[:min_len]
    rationale = rationale[:min_len]
    
    # Recompute k after potential truncation
    k = int(rationale.sum())
    if k == 0:
        return None
    
    # Binarize: top-k positions by attribution magnitude
    k = min(k, len(word_attributions))  # Safety clamp
    top_k_idx = np.argsort(word_attributions)[-k:]
    pred_rationale = np.zeros(min_len)
    pred_rationale[top_k_idx] = 1
    
    precision = precision_score(rationale, pred_rationale, zero_division=0)
    recall = recall_score(rationale, pred_rationale, zero_division=0)
    f1 = f1_score(rationale, pred_rationale, zero_division=0)
    
    return {'precision': precision, 'recall': recall, 'f1': f1}

