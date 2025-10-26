# Data Card: eRisk Dataset

## 1. Dataset Overview

- **Name:** eRisk 2025 – Early Risk Prediction on the Internet (Depression Task)
- **Source:** [CLEF eRisk Shared Task](https://erisk.irlab.org/)
- **Purpose:** The eRisk dataset is designed to support research on early detection of mental-health and behavioral risks from users’ online text data.
- **Format:** TREC-style XML files (converted to `.parquet` after cleaning).
- **Language:** English
- **Task Type:** Binary classification (`depressed` vs. `control`)
- **Intended Use:** Research on early detection of mental health conditions using NLP.

---

## 2. Data Composition

| Attribute      | Description                                       |
| -------------- | ------------------------------------------------- |
| **user_id**    | Unique anonymized identifier for each Reddit user |
| **label**      | `1` = depressed, `0` = control                    |
| **text**       | Cleaned user post text                            |
| **num_tokens** | Number of tokens in each post                     |
| **pre**        | Pre-text metadata (from original TREC)            |

- Total posts (after cleaning): 10,383
- Total unique users: 2,840
- Average posts per user: 3.66
- Average tokens per post: 11.47
- Min / Max tokens per post: 1 / 326

---

## 3. Collection Process

- Data collected from Reddit as part of the CLEF eRisk shared tasks.
- Posts grouped per user for early risk prediction.
- Labels manually annotated by eRisk organizers.
- eRisk 2025 focuses on **contextual and conversational approaches** to depression detection.

---

## 4. Preprocessing & Cleaning

Performed in `scripts/clean_eRisk.py`:

1. Parse `.trec` files into structured `DataFrame`.
2. Normalize text (lowercase, remove HTML, emojis, URLs, and special characters).
3. Optionally filter for English posts.
4. Remove very short posts based on token count.
5. Save cleaned output to `data/processed/eRisk_clean.parquet`.

---

## 5. Data Splits

Performed via `scripts/split_datasets.py` (grouped by user):

- **Train:** 70%
- **Test:** 30%

---

## 6. Intended Uses

- Training and evaluating NLP models for early depression detection.
- Studying linguistic markers of depression in online language.
- Experimenting with contextual or conversational models (per eRisk 2025 guidelines).

---

## 7. Limitations

- Labels inferred or self-reported — may not reflect formal clinical diagnoses.
- English-only data excludes non-English users.
- Possible temporal bias due to collection period.

---

## 8. Ethical Considerations

- Do not use for diagnosis or individual intervention.
- Respect user anonymity; never attempt re-identification.
- Use only for approved, non-commercial research under eRisk user agreements.
- Follow CLEF eRisk ethical guidelines and data usage terms.

---

## 9. License

- eRisk 2025 collections are available for research purposes only under proper user agreements.
- Redistribution or commercial use is strictly prohibited.
- Access requires agreement to the official eRisk data license.

---

## 10. Citation

1. **Crestani, F., Losada, D., & Parapar, J. (2022).**  
   _Early Detection of Mental Health Disorders by Social Media Monitoring._  
   Studies in Computational Intelligence, 1018, 4.

2. **Parapar, J., Perez, A., Wang, X., & Crestani, F. (2025).**  
   _eRisk 2025: Contextual and Conversational Approaches for Depression Challenges._  
   In _European Conference on Information Retrieval (pp. 416–424)._

---

## 11. More Info

For dataset access and license details, visit the official website:  
[https://erisk.irlab.org/](https://erisk.irlab.org/)
