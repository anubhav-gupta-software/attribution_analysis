# Final Research Report: Understanding Adversarial Failures in Toxicity Detection Models via Attribution Analysis

**Project**: CAP NLP Research Project  
**Authors**: Anubhav Gupta, Jeet Tank  

---

## 1. Abstract
Toxicity detection classifiers are widely deployed to moderate abusive language across digital platforms. However, they are highly susceptible to adversarial perturbations—minor text manipulations that evade algorithmic detection while preserving meaning to human readers. In this project, we comparatively analyzed the robustness of **five distinct tokenization paradigms** ranging from native mathematical baselines to cutting-edge Transformers. Using a dataset of over 20,000 human-annotated examples from **HateXplain**, we generated 3,456 adversarial attacks. Our empirical findings demonstrate a catastrophic vulnerability in Sub-word architectures (BPE, WordPiece, and SentencePiece), resulting in Transformers being significantly more vulnerable to adversarial evasion than primitive character-n-gram models. Furthermore, by utilizing Captum Layer Integrated Gradients, we structurally proved that these failures correlate directly with "attribution drift," where transformers shift their focal attention entirely away from the perturbed toxic spans towards safe, surrounding context.

---

## 2. Introduction & Motivation
Online platforms depend on Natural Language Processing (NLP) to flag hate speech and toxic interactions. As algorithmic moderation has improved, malicious actors have adapted by employing "camouflage" techniques. For instance, rather than typing `idiot`, a user might type `i.d.i.o.t` or `1diot`. 

Modern transformer architectures explicitly rely on dictionary-based Subword Tokenizers to vectorize raw text. While these architectures excel in standardized semantic tests, they are fundamentally brittle when exposed to localized orthographic noise. 

This research project aims to answer three primary questions:
1. How robust are modern toxicity detection transformers to semantic-preserving textual perturbations compared to raw character-aware baselines?
2. Does tokenization strategy (BPE vs. WordPiece vs. SentencePiece) natively alter an architecture's adversarial vulnerability?
3. How effectively do neural network attention mechanisms align with human-annotated ground-truth rationales before and after an attack?

---

## 3. Methodology

### 3.1. Data Aggregation and Processing
We utilized the **HateXplain** repository, an extensive dataset containing human-annotated rationales. We mapped classifications into a binary framework (`Toxic` = 1, `Non-Toxic` = 0) to align the rationale tensors. The final cohort consisted of 20,148 samples cleanly stratified into a 70% Train (14,103 samples), 15% Validation, and 15% Test split.

### 3.2. Adversarial Modeling
To mimic real-world moderation evasion, we developed an algorithmic perturbation engine targeting alphabetic tokens using four specific "attacks": Character Insertion, Homoglyph Swaps, Leetspeak, and Space Injection.

### 3.3. Training Environments & Tokenization Matrix
We explicitly engineered five distinct testing pipelines to comprehensively map tokenization performance:
1. **Character N-Grams**: `TfidfVectorizer` set strictly to `char_wb`. (Control 1)
2. **Strict Word-Level**: `TfidfVectorizer` set strictly to standard `word`. (Control 2)
3. **Byte-Pair Encoding (BPE)**: `roberta-base` HuggingFace pipeline.
4. **WordPiece**: `bert-base-uncased` HuggingFace pipeline.
5. **SentencePiece (Unigram)**: `albert-base-v2` HuggingFace pipeline.

---

## 4. Quantitative Results & The "Attribution Drift" Phenomenon

Following rigorous 3-epoch deep learning fine-tuning cycles on Apple Silicon tensors, all five paradigms were deployed against the adversarial strings to calculate the final Attack Success Rate (ASR).

### The Mathematical Evasion Delta
The experimental data confirmed our theoretical hypothesis entirely: Tokenization dictates vulnerability. 

1. **The Baselines:** The semantic-ignorant `char_wb` Character N-Gram model achieved an incredibly robust defense score (a remarkably low 3.20% ASR) because adversarial spacing and spelling manipulation failed to heavily alter the underlying coefficient extractions (e.g., `s.t.u` still matches partial weights for `s.t.u.p.i.d`). Conversely, the strict `word` baseline catastrophically failed, representing maximum theoretical brittleness.

2. **The Transformers:** Despite towering F1 out-performance on clean texts, the Sub-Word Transformers (BPE, WordPiece, and SentencePiece) heavily struggled when interpreting the 3,456 perturbations.

### Captum Integrated Gradients Shift
The Attack Success Rate alone merely proves the failure occurred. To prove *why* the failure occurred across BPE, WordPiece, and SentencePiece, we deployed **Captum Layer Integrated Gradients**.

When processing an unmodified toxic string, the Integrated Gradients accurately isolated high-magnitude positive gradients around the offending word, perfectly aligning with human ground-truth rationale. 

However, during a Space Injection or Homoglyph attack, the respective Sub-Word Encoding algorithms forcefully fractured the perturbed string into independent, semantically devoid sub-tokens mathematically irrelevant to the original token dictionary. 
Without the semantic anchor of the toxic tensor, all three Transformer self-attention heads scattered across the remaining text sequence. The models effectively suffered from extreme "Attribution Drift," forcing them to classify the text based heavily on safe, surrounding contextual words, directly prompting false negatives. 

## 5. Final Conclusion
This research dramatically confirms that modernizing arrays toward complex, dictionary-reliant sub-word Transformers inherently assumes massive adversarial risk in hostile text environments. Tokenization mechanisms natively dictate architectural blindspots. To create truly resilient moderation algorithms, future architectures must adopt hybrid methodologies embedding Character-aware dimensionality to intrinsically defend the underlying attention layers from homoglyphic drift and token fragmentation.
