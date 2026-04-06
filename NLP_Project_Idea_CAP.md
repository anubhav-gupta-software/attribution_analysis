**Understanding Adversarial Failures in Toxicity Detection Models via Attribution Analysis**

Anubhav Gupta, Jeet Tank

1\. **Introduction and Relevance to NLP**  
Toxicity detection is a common Natural Language Processing (NLP) task used to identify harmful or abusive language in online text. Many online platforms rely on automated systems to detect toxic comments and moderate content at scale using machine learning models.

However, these systems can fail when the text is slightly modified. For example, users may change spelling or insert punctuation (e.g., idiot to i.d.i.o.t) to bypass toxicity filters while keeping the meaning the same. This raises questions about the robustness of NLP models to adversarial or intentionally modified inputs.

In this project, we will study how toxicity detection models behave when toxic text is modified using small adversarial changes such as character substitutions or punctuation insertion. We will train two classifiers: a baseline Logistic Regression model with TF-IDF features and a RoBERTa-based transformer model, and evaluate how their performance changes under adversarial perturbations.

To understand model behavior, we will apply Integrated Gradients to identify which tokens contribute most to toxicity predictions. We will then compare these token attributions with human rationale annotations from the HateXplain dataset to analyze why adversarial modifications cause models to fail.

This project is relevant to NLP because it studies text classification, adversarial robustness, and model interpretability, which are active research areas in modern NLP systems.

2\. **Current State of the Research Literature**  
Toxicity detection has been widely studied in NLP because of its importance for moderating harmful language on online platforms. Many modern systems rely on supervised learning models trained on labeled datasets to classify whether a comment is toxic or abusive. Recent work has also explored adversarial attacks and explainability methods to understand how these models behave and why they fail. In this project, we build on prior work in toxicity detection, adversarial robustness, and model interpretability.

The following papers are most relevant to this project:

1. HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection: [https://arxiv.org/abs/2012.10289](https://arxiv.org/abs/2012.10289) : This paper introduces the HateXplain dataset which includes toxicity labels and human-annotated rationale spans highlighting tokens responsible for the label.  
2. TextBugger: Generating Adversarial Text Against Real-world Applications: [https://arxiv.org/abs/1812.05271](https://arxiv.org/abs/1812.05271) : This paper shows that NLP models are vulnerable to adversarial attacks created through small character-level perturbations such as spelling changes or punctuation insertion. These perturbations can cause text classifiers to misclassify toxic content while keeping the meaning understandable to human readers.  
3. GRAINS: Gradient-based Attribution for Inference-Time Steering of LLMs and VLMs: [https://arxiv.org/abs/2507.18043](https://arxiv.org/abs/2507.18043) : This paper introduces a gradient-based attribution method that uses Integrated Gradients to identify influential tokens in model predictions. It shows how token-level attribution signals can be used to analyze and steer model behavior.

These works motivate our project, which studies how adversarial perturbations affect toxicity detection models and analyzes how the tokens influencing model predictions change under adversarial inputs.

3\. **Research Questions**  
This project aims to study how adversarial perturbations affect toxicity detection models and how these perturbations influence the tokens that models rely on for their predictions. The following research questions guide our study:

* How robust are toxicity detection models to adversarial text perturbations? Specifically, we examine whether small character-level modifications such as spelling changes or punctuation insertion reduce the ability of models to correctly classify toxic comments.  
* Which tokens do toxicity detection models rely on when making predictions? Using Integrated Gradients, we analyze the token-level attribution scores to identify which parts of the text most strongly influence model predictions.  
* Do adversarial perturbations change the tokens that models consider important? We study whether adversarial modifications cause models to shift attention away from the key tokens that originally contributed to the toxicity prediction.  
* How well do model attribution signals align with human rationales? Using the rationale annotations from the HateXplain dataset, we compare the tokens identified by the model with the tokens highlighted by human annotators.

4\. **Experimental Design and Evaluation Metrics**  
To answer the research questions, we will evaluate both the robustness and interpretability of toxicity detection models.

We will use the HateXplain dataset, which contains labeled toxic and non-toxic comments along with human-annotated rationale spans indicating which tokens justify the label. These rationales will allow us to compare model explanations with human annotations.

We will train two models: a Logistic Regression classifier with TF-IDF features as a baseline and a RoBERTa-based classifier representing a modern transformer model. After training, we will generate adversarial examples by applying small character-level perturbations such as spelling changes and punctuation insertion to toxic comments.

To analyze model reasoning, we will apply Integrated Gradients to compute token-level attribution scores and identify which tokens influence the model’s predictions. These attributions will then be compared with the human rationale spans in HateXplain.

We will evaluate performance using standard classification metrics including accuracy, precision, recall, and F1 score on both the original and adversarial datasets. We will also measure the performance drop under adversarial perturbations to quantify model robustness and analyze the overlap between attribution scores and human rationales to evaluate interpretability.  
