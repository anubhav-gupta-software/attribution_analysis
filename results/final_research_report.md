# Final Research Report: Understanding Adversarial Failures in Toxicity Detection Models via Attribution Analysis

**Project**: CAP NLP Research Project  
**Authors**: Anubhav Gupta, Jeet Tank  

---

## 1. Abstract
Toxicity detection classifiers are widely deployed to moderate abusive language across digital platforms. However, they are highly susceptible to adversarial perturbations—minor text manipulations that evade algorithmic detection while preserving meaning to human readers. In this project, we comparatively analyzed the robustness of two distinct architectural paradigms: a Character N-Gram Logistic Regression baseline and a state-of-the-art RoBERTa Transformer model. Using a dataset of over 20,000 human-annotated examples from **HateXplain**, we generated 3,456 adversarial attacks. Our empirical findings demonstrate a catastrophic vulnerability in Sub-word (BPE) tokenizers, resulting in the RoBERTa model being **10.8x more vulnerable** to adversarial evasion than the primitive character-n-gram model (34.71% vs 3.20% Attack Success Rate). Furthermore, by utilizing Captum Layer Integrated Gradients, we structurally proved that these failures correlate directly with "attribution drift," where the transformer shifts its focal attention entirely away from the perturbed toxic spans towards safe, surrounding context.

---

## 2. Introduction & Motivation
Online platforms depend on Natural Language Processing (NLP) to flag hate speech and toxic interactions. As algorithmic moderation has improved, malicious actors have adapted by employing "camouflage" techniques. For instance, rather than typing `idiot`, a user might type `i.d.i.o.t` or `1diot`. 

Modern transformer architectures explicitly rely on dictionary-based Subword Tokenizers, such as Byte-Pair Encoding (BPE) or WordPiece, to vectorize raw text. While these architectures excel in standardized semantic tests, they are fundamentally brittle when exposed to localized orthographic noise. 

This research project aims to answer three primary questions:
1. How robust are modern toxicity detection transformers to semantic-preserving textual perturbations compared to raw character-aware baselines?
2. When adversarial texts successfully evade detection, *why* do they fail?
3. How effectively do the neural network's attention mechanisms align with human-annotated ground-truth rationales before and after an attack?

---

## 3. Methodology

### 3.1. Data Aggregation and Processing
We utilized the **HateXplain** repository, an extensive multi-aspect hate speech dataset containing ternary labels (`normal`, `offensive`, `hate speech`). 
We mapped classifications into a binary framework (`Toxic` = 1, `Non-Toxic` = 0) by taking the majority vote across annotators. Crucially, the dataset includes human-annotated rationales (the precise tokens humans flagged as toxic). We calculated a Boolean threshold vote to generate ground-truth alignment tensors for every data point. The final cohort consisted of 20,148 samples, stratified into a 70% Train, 15% Validation, and 15% Test split.

### 3.2. Adversarial Modeling
To mimic real-world moderation evasion, we developed an algorithmic perturbation engine targeting alphabetic tokens. We deployed four specific "attacks":
- **Character Insertion**: Injecting non-alphanumeric noise (e.g., `racist` -> `r.a.c.i.s.t`).
- **Homoglyph Swap**: Swapping similar typographical symbols (e.g., `hate` -> `h@te`).
- **Leetspeak**: Converting alphabetic characters to integers (e.g., `stupid` -> `5tup1d`).
- **Space Injection**: Uniformly spacing discrete strings (e.g., `nazi` -> `n a z i`).
The engine operated at a 30% severity threshold, selectively disrupting targeted spans.

### 3.3. Training Environments
1. **Baseline Model**: We deployed a Logistic Regression model anchored by a `TfidfVectorizer` explicitly tuned for character `n-grams` within word boundaries (`char_wb`). 
2. **Transformer Model**: We instantiated the `roberta-base` architecture via the HuggingFace `Trainer` API, executing a 3-epoch deep learning fine-tuning sequence processing against Apple Silicon (`mps` tensor acceleration), optimized by AdamW weight-decay parameters. 

---

## 4. Quantitative Results and Findings

Following training across the primary sequence, both models were tested against the control test-split, producing the following internal classification baselines on unmodified strings:
- **Baseline LR (F1):** `0.495`
- **RoBERTa (F1):** `~0.680` (Standard Transformer outperformance on clean data).

### 4.1. Adversarial Attack Success Rate (ASR)
The models were then fed the generated batch of 3,456 perturbed toxic strings. The Attack Success Rate (ASR) measures the percentage of originally detected toxic statements that slipped past the filter and were misclassified as 'safe':

| Architecture Type | Tokenization Vector | Attack Success Rate (ASR) |
| :--- | :--- | :--- |
| **Logistic Regression TF-IDF** | Character N-Grams | **3.20%** |
| **RoBERTa Transformer** | Sub-word Encodings (BPE) | **34.71%** |

**Conclusion of Data:** The RoBERTa Transformer demonstrated catastrophic failure rates against minor manipulations. It was over 1,000% (10x) more vulnerable to evasion techniques than the primitive TF-IDF model.

---

## 5. Interpretability: Why Did the Transformer Fail?

The ASR delta alone merely proves that the failure occurred. To prove *why* the failure occurred, we utilized an attribution proxy. 

For the Logistic Regression model, we approximated token-level logic by applying its coefficient weights linearly over its extracted `char_wb` n-grams. For RoBERTa, we implemented **Captum Layer Integrated Gradients**. By integrating across the network’s embedding interpolation baseline, we generated mathematically rigorous alignment scores demonstrating exactly how the Transformer prioritized its context windows.

### 5.1. The "Attribution Drift" Phenomenon
When processing an unmodified token string containing an abusive term, the Integrated Gradients accurately isolated high-magnitude positive gradients around the offending word. This perfectly intersected with the HateXplain human-ground-truth rationale (High IOU).

However, during a Space Injection or Homoglyph attack, RoBERTa's Byte-Pair Encoding algorithm forcefully fractured the perturbed string into independent, semantically devoid sub-tokens. 
- *Original Sequence:* `[..., 'you', 'are', 'stupid', ...]` -> Model assigns +0.8 Attribution to `stupid`. 
- *Adversarial Sequence:* `[..., 'you', 'are', 's.t.u.p.i.d', ...]` -> The tokenizer generates arbitrary independent tokens.

Without the semantic anchor of the toxic tensor in its dictionary, the self-attention heads scattered across the remaining sequence, drastically reducing its Intersection Over Union (IOU) with human rationale. The model effectively suffered from "Attribution Drift," forcing it to classify the text based on safe, surrounding contextual words, prompting a false negative.

Conversely, the Baseline LR model naturally ignored the disruptions because the `char_wb` tokenizer intrinsically generated sub-strings (e.g., `s.t` or `p.i`) that retained partial toxic coefficients regardless of the dictionary.

---

## 6. Final Conclusion

This research confirms that moving toward strictly semantic-based, dictionary-reliant Transformers like RoBERTa fundamentally degrades adversarial safety in hostile text environments. While language models excel at holistic contextual understanding, their Subword Tokenization algorithms are an architectural liability for toxicity detection. 

To create truly resilient content moderation algorithms, future architectures must hybridize Subword embeddings with Character-aware convolutions, inherently protecting the attention layers from homoglyphic masking and typographical fragmentation. 
