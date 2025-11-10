# Detecting-and-Mitigating-Hallucinations-in-LLM-Generated-Medical-Summaries

## Project description

Large Language Models (LLMs) such as GPT-4, LLaMA-3, and Mistral are increasingly used to summarize medical reports and research abstracts. However, these models frequently hallucinate i.e., they generate false or unsupported medical statements, such as incorrect drug dosages, non-existent clinical trials, or misattributed findings. In the medical domain, such errors can have serious implications for patient safety and trust in AI-assisted decision-making.
The goal of this project is to develop a system that automatically detects and mitigates hallucinations in LLM-generated medical summaries. Specifically, we will evaluate whether each sentence in a generated summary is supported, contradicted, or unsupported by the original source text.
Our NLP task is natural language inference (NLI) and sentence-level factual consistency classification. To achieve this, we will:
1.	Generate summaries of real medical texts using domain-specific LLMs such as BioGPT or Med-PaLM.
2.	Fine-tune transformer-based models like BioBERT and DeBERTa-v3 on labeled NLI datasets (e.g., MedNLI, PubMedQA) to classify factual alignment between source–summary pairs.
3.	Compute cosine similarity between contextual embeddings (e.g., using Sentence-BERT) as an auxiliary check for semantic consistency.
4.	Optionally, include an LLM-as-a-Judge module where a secondary LLM evaluates hallucination severity in natural-language explanations.


## Dataset:
We plan to use publicly available datasets such as PubMedQA, MedNLI, and PubMed abstracts. The MedNLI dataset contains ~14,000 sentence pairs labeled as entailment, contradiction, or neutral, providing ground truth for training and evaluation. Additionally, we will generate LLM-based summaries of PubMed abstracts and manually annotate a small subset for hallucination detection fine-tuning.

## Models and Training Plan:
Our baseline will be a simple cosine-similarity-based model using Sentence-BERT embeddings. We will then fine-tune BioBERT and DeBERTa-v3-base on the MedNLI dataset for the NLI classification task. The data will be split into 80% training, 10% validation, and 10% test sets. Fine-tuning will be done using standard cross-entropy loss with early stopping based on validation F1.

## Evaluation Metrics:
Model quality will be assessed using Precision, Recall, F1-Score, and Accuracy for hallucination detection. For embedding-based methods, we will also report cosine similarity thresholds and AUC-ROC to measure discriminative ability.

## Related Work:
Recent studies have addressed factuality and hallucination detection in medical text generation. We will reference the following key works:
1.	Factuality Evaluation for Plain Language Summarization of Medical Evidence (Joseph et al., ACL 2024) (Joseph et al., ACL 2024) introduces a benchmark for detecting factual errors in plain-language summaries of medical evidence.
2.	Fact-Controlled Diagnosis of Hallucinations in Medical Text Summarization (Suhas et al., 2025) shows that general-domain detectors fail on medical data, motivating specialized models like ours.
3.	Factuality Evaluation of Summarization Based on Natural Language Inference and Claim Extraction (Scirè et al., Findings of ACL 2024) uses claim extraction and NLI for factuality evaluation, closely aligning with our entailment-based approach.
These papers form the foundation for our project, which extends factual consistency detection to LLM-generated medical summaries using transformer-based NLI and embedding-similarity techniques.

