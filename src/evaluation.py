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
