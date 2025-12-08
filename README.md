# Depression-Risk Detection Using Machine Learning and Transformer Models
This repository presents an end-to-end pipeline for detecting potential indicators of depression in short, informal text messages. The project investigates how traditional machine-learning methods compare to modern transformer-based models, and examines the feasibility, challenges, and ethical considerations involved in applying NLP to mental-health–related signals.

## Overview
Depression can manifest through subtle linguistic patterns—particularly in online spaces where communication tends to be brief, informal, and context-poor. Automatically identifying these signals is a difficult modeling problem, but one with meaningful potential impact in digital well-being tools, safety systems, and early-warning research.
This project evaluates several modeling approaches on the eRisk 2025 dataset (10,383 short messages labeled as depressed vs. not depressed). The dataset is highly imbalanced and includes noisy, informal text, making it a realistic test for model robustness.

Our pipeline includes:
- Text preprocessing and dataset curation
- Baseline classical ML models using TF-IDF
- Fine-tuning a BERT classifier for contextual modeling
- Comprehensive evaluation using macro-F1 and class-specific precision/recall
- Ethical framing around mental-health detection

## Key Features
### 1. Data Preparation & Cleaning
We preprocess eRisk messages by removing duplicates, normalizing text, stripping URLs/emojis, and generating train/test splits.
The dataset exhibits a ~70/30 class imbalance, so macro-F1 is used as the primary evaluation metric.

### 2. Baseline Models
We train and evaluate two classical models using TF-IDF features:
- Logistic Regression
- Support Vector Machine (SVM)

SVM serves as a strong baseline, achieving:
- Accuracy: 0.702
- Macro-F1: 0.666

These results demonstrate that simple linear models remain competitive on sparse text.

###  3. Transformer-Based Model (BERT)
We fine-tune a BERT base model using HuggingFace Transformers with:
- Learning rate: 2e-5
- Batch size: 16
- 5 epochs
- AdamW optimizer + linear warmup

A non-fine-tuned BERT model performs poorly due to domain mismatch, but fine-tuning dramatically improves performance, achieving:
- Accuracy: 0.724
- Macro-F1: 0.679

This highlights the importance of domain-specific adaptation for transformer models.

## Results Summary
| Model	                        | Accuracy	| Macro-F1    |
| ------------------------------|-----------|------------ |
| TF-IDF + Logistic Regression	| 0.659	    | 0.629       |
| TF-IDF + SVM	                | 0.702	    | 0.666       |
| BERT (not fine-tuned)	        | 0.687	    | 0.411       |
| BERT (fine-tuned)	            | 0.724	    | 0.679       |

Fine-tuning allows BERT to surpass classical baselines—confirming that contextual modeling benefits this task only when adapted to the training domain.

## Limitations
Despite promising results, several factors constrain real-world reliability:
- Dataset imbalance skews predictions toward the majority class.
- Very short, noisy messages (avg. ~11 tokens) limit the available linguistic signal.
- Depression is a complex, long-term condition that cannot be inferred reliably from isolated posts.
- Single-run training due to time and resource constraints may underrepresent the full potential of the models.
  
These limitations highlight both technical and conceptual challenges in mental-health NLP research.

## Future Directions
Several extensions could substantially strengthen this line of work:
- Sequence-level modeling: analyzing sets of messages from the same user rather than isolated posts.
- Richer datasets with user-level continuity, enabling temporal modeling of linguistic patterns.
- Improved training regimes (multiple seeds, hyperparameter sweeps).
- Exploration of safety-aware deployment, should the system ever be integrated into real-time environments (e.g., chat platforms).
  
While earlier discussions considered integration into platforms like Discord, practical deployment would require stringent privacy protections, human oversight, and ethical safeguards.

## Ethical Considerations
Any work involving mental-health inference must prioritize safety and responsible use. Misclassifications—especially false negatives—carry serious implications. This project is intended for research exploration only and not for clinical or diagnostic use.
