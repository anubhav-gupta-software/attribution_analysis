import torch
import numpy as np
from models.roberta import RobertaModel
from models.lr_tfidf import LRTfidfModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from captum.attr import IntegratedGradients
from captum.attr import LayerIntegratedGradients

class RobertaAttribution:
    def __init__(self, model_dir='models/roberta'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        # Look at the embeddings layer
        self.lig = LayerIntegratedGradients(self._forward_wrapper, self.model.roberta.embeddings)
        
    def _forward_wrapper(self, input_ids, attention_mask):
        # wrapper for captum
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return logits
        
    def get_attribution(self, text, target_class=1):
        """
        Calculates Integrated Gradients for a given text.
        Returns token attributions mapped back to exact input words.
        """
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # We need a baseline to compute IG. The pad token is a good baseline.
        pad_token_id = self.tokenizer.pad_token_id
        ref_input_ids = torch.full_like(input_ids, pad_token_id).to(self.device)
        
        # Calculate attributions
        attributions, delta = self.lig.attribute(
            inputs=(input_ids,),
            baselines=(ref_input_ids,),
            additional_forward_args=(attention_mask,),
            target=target_class,
            return_convergence_delta=True,
            internal_batch_size=16
        )
        
        # attributions are of shape (1, seq_len, embed_dim)
        # We sum over embed_dim to get score per token
        attributions_sum = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
        
        # Return lists
        return tokens, attributions_sum
        
class LRAttribution:
    def __init__(self, model_dir='models/lr'):
        self.model = LRTfidfModel(model_dir=model_dir)
        self.model.load()
        self.pipeline = self.model.pipeline
        self.tfidf = self.pipeline.named_steps['tfidf']
        self.lr = self.pipeline.named_steps['lr']
        self.feature_names = self.tfidf.get_feature_names_out()
        # Coeffs for target class (assuming toxic is class 1)
        self.coef = self.lr.coef_[0]
        
    def get_attribution(self, text):
        """
        Returns feature importance for the given text.
        (TF-IDF value * coefficient)
        """
        vec = self.tfidf.transform([text]).toarray()[0]
        attributions = vec * self.coef
        
        # Map back to features
        nonzero_idx = vec.nonzero()[0]
        
        # We want token-level attributions, but TF-IDF is char-ngram. 
        # So we approximate by distributing char-ngram attribution across tokens.
        words = text.split()
        word_attributions = np.zeros(len(words))
        
        for idx in nonzero_idx:
            feat_name = self.feature_names[idx]
            attr_val = attributions[idx]
            # Naive mapping: if the char-ngram is present in the word, add attribution
            # This is a very rough approximation, but standard for linear + char ngrams.
            for i, w in enumerate(words):
                if feat_name in w:
                    word_attributions[i] += attr_val
                    
        return words, word_attributions

class BertAttribution:
    def __init__(self, model_dir='models/bert'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.lig = LayerIntegratedGradients(self._forward_wrapper, self.model.bert.embeddings.word_embeddings)
        
    def _forward_wrapper(self, input_ids, attention_mask):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return logits
        
    def get_attribution(self, text, target_class=1):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        pad_token_id = self.tokenizer.pad_token_id
        ref_input_ids = torch.full_like(input_ids, pad_token_id).to(self.device)
        
        attributions, delta = self.lig.attribute(
            inputs=(input_ids,),
            baselines=(ref_input_ids,),
            additional_forward_args=(attention_mask,),
            target=target_class,
            return_convergence_delta=True,
            internal_batch_size=16
        )
        
        attributions_sum = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
        return tokens, attributions_sum

class AlbertAttribution:
    def __init__(self, model_dir='models/albert'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.lig = LayerIntegratedGradients(self._forward_wrapper, self.model.albert.embeddings.word_embeddings)
        
    def _forward_wrapper(self, input_ids, attention_mask):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return logits
        
    def get_attribution(self, text, target_class=1):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        pad_token_id = self.tokenizer.pad_token_id
        ref_input_ids = torch.full_like(input_ids, pad_token_id).to(self.device)
        
        attributions, delta = self.lig.attribute(
            inputs=(input_ids,),
            baselines=(ref_input_ids,),
            additional_forward_args=(attention_mask,),
            target=target_class,
            return_convergence_delta=True,
            internal_batch_size=16
        )
        
        attributions_sum = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
        return tokens, attributions_sum

if __name__ == "__main__":
    pass
