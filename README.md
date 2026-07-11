# Adversarial Robustness of NLP-Based Phishing Email Classifiers

**A Multi-Dataset Empirical Study**

Submitted to the Columbia Junior Science Journal (CJSJ) 2026
Author: Muktadir Arif · Supervisor: Dr. Latif Siddiq

---

## Overview

This repository contains the code, results, and figures for an empirical study comparing how **character-level (homoglyph substitution)** and **word-level (synonym replacement)** adversarial text perturbations degrade the classification accuracy of a fine-tuned **DistilBERT** phishing/spam detector, evaluated across **five benchmark datasets** spanning corporate email, general email, SMS, and phishing-focused corpora.

Trained model checkpoints are hosted separately on the Hugging Face Hub (linked below) due to file size; this repository holds the code, CSV results, and publication figures.

---

## Key Findings

1. **Homoglyph attack severity scales with message length.** Email-heavy datasets (Enron Spam, Phishing Email Dataset, PhishingEmailDetection) suffered severe accuracy degradation (−19 to −30 percentage points at 30% perturbation), while short-text datasets (SMS Spam, Spam Detection) degraded only −1 to −2 pp.

2. **Two distinct failure modes emerge under homoglyph attack:**
   - **Precision collapse** (email datasets) — false positives explode (e.g., Enron Spam: 10 → 2,070 FPs) while false negatives stay near zero.
   - **Recall collapse** (SMS Spam) — precision stays high, but recall drops as spam messages evade detection.

3. **Synonym replacement is consistently negligible** — across all 5 datasets and all 3 perturbation rates (10/20/30%), accuracy degradation never exceeded ~1.5 percentage points.

All homoglyph-attack conditions (except SMS Spam at 10%) were statistically significant after Benjamini-Hochberg FDR correction (bootstrap resampling, n = 1,000; p-corrected < 0.001).

---

## Repository Structure

```
.
├── code/
│   ├── train_model.py              # DistilBERT fine-tuning per dataset
│   ├── homoglyph_attack.py         # Character-level substitution attack
│   ├── synonym_attack.py           # WordNet-based synonym replacement attack
│   ├── run_experiment.py           # Full attack + evaluation pipeline
│   ├── statistics.py               # Bootstrap resampling + BH-FDR correction
│   └── generate_figures.py         # All 8 publication figures
├── results/
│   ├── attack_results_*.csv        # Per-dataset attack results (5 files)
│   ├── stats_*.csv                 # Per-dataset statistical test results (5 files)
│   ├── master_attack_results.csv   # Combined results, all datasets
│   ├── master_statistical_results.csv
│   └── paper_summary_table.csv
├── figures/
│   ├── fig1_baseline_summary.png
│   ├── fig2_accuracy_synonym.png
│   ├── fig3_accuracy_homoglyph.png
│   ├── fig4_precision_synonym.png
│   ├── fig5_precision_homoglyph.png
│   ├── fig6_heatmap_summary.png
│   ├── fig7_false_positives_homoglyph.png
│   └── fig8_false_positives_synonym.png
└── README.md
```

---

## Datasets

| # | Dataset | HuggingFace ID | Size | Domain |
|---|---|---|---|---|
| 1 | Enron Spam | [`SetFit/enron_spam`](https://huggingface.co/datasets/SetFit/enron_spam) | 33,716 emails | Corporate email |
| 2 | Spam Detection | [`Deysi/spam-detection-dataset`](https://huggingface.co/datasets/Deysi/spam-detection-dataset) | 10,900 emails | General email |
| 3 | SMS Spam | [`ucirvine/sms_spam`](https://huggingface.co/datasets/ucirvine/sms_spam) | 5,574 messages | SMS/text |
| 4 | Phishing Email Dataset | [`zefang-liu/phishing-email-dataset`](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset) | 18,650 emails | Phishing-focused |
| 5 | PhishingEmail Detection | [`cybersectony/PhishingEmailDetectionv2.0`](https://huggingface.co/datasets/cybersectony/PhishingEmailDetectionv2.0) | 20,000 (sampled) | Multi-source phishing |

> Dataset 5 originally contains 200,000 rows across 4 classes; only email-labeled rows were retained and sampled to 20,000 to fit within Colab's free-tier RAM limits.

---

## Trained Models

Fine-tuned `distilbert-base-uncased` checkpoints (one per dataset) are hosted on the Hugging Face Hub:

| Dataset | Model Repo |
|---|---|
| Enron Spam | [`[your-hf-username]/distilbert-phishing-enron-spam`](https://huggingface.co/[your-hf-username]/distilbert-phishing-enron-spam) |
| Spam Detection | [`[your-hf-username]/distilbert-phishing-spam-detection`](https://huggingface.co/[your-hf-username]/distilbert-phishing-spam-detection) |
| SMS Spam | [`[your-hf-username]/distilbert-phishing-sms-spam`](https://huggingface.co/[your-hf-username]/distilbert-phishing-sms-spam) |
| Phishing Email Dataset | [`[your-hf-username]/distilbert-phishing-email-dataset`](https://huggingface.co/[your-hf-username]/distilbert-phishing-email-dataset) |
| PhishingEmail Detection | [`[your-hf-username]/distilbert-phishing-detection-v2`](https://huggingface.co/[your-hf-username]/distilbert-phishing-detection-v2) |

Load any model directly with `transformers`:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "[your-hf-username]/distilbert-phishing-enron-spam"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

---

## Methodology Summary

| Parameter | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Training | 3 epochs, batch size 32, AdamW, lr = 2e-5, seed = 42 |
| Hardware | Google Colab, T4 GPU |
| Attack 1 | Homoglyph substitution (10-character Latin → Cyrillic/Greek Unicode mapping) |
| Attack 2 | Synonym replacement via NLTK WordNet |
| Perturbation rates | 10%, 20%, 30% |
| Train/test split | 80/20, stratified |
| Statistics | Bootstrap resampling (n = 1,000), one-sample t-tests, Benjamini-Hochberg FDR correction |

### Homoglyph substitution map

```python
HOMOGLYPH_MAP = {
    'a': 'а',  # Cyrillic а (U+0430)
    'e': 'е',  # Cyrillic е (U+0435)
    'o': 'о',  # Cyrillic о (U+043E)
    'p': 'р',  # Cyrillic р (U+0440)
    'c': 'с',  # Cyrillic с (U+0441)
    'x': 'х',  # Cyrillic х (U+0445)
    'i': 'і',  # Cyrillic і (U+0456)
    'u': 'υ',  # Greek upsilon (U+03C5)
    'y': 'у',  # Cyrillic у (U+0443)
    'k': 'к',  # Cyrillic к (U+043A)
}
```

---

## Reproducing the Results

```bash
# 1. Install dependencies
pip install --upgrade transformers accelerate datasets
pip install scikit-learn pandas matplotlib seaborn nltk scipy statsmodels

# 2. Fine-tune a model on a chosen dataset
python code/train_model.py --dataset enron_spam

# 3. Run the full attack + evaluation pipeline
python code/run_experiment.py --dataset enron_spam

# 4. Regenerate all figures from results CSVs
python code/generate_figures.py
```

This codebase was developed and run on Google Colab (T4 GPU); paths and Drive-mounting logic in the scripts may need adjustment for local or other cloud environments.

---

## Figures

| Figure | Description |
|---|---|
| Fig. 1 | Baseline (unattacked) accuracy and precision across all five datasets |
| Fig. 2 | Accuracy vs. perturbation rate — Synonym substitution |
| Fig. 3 | Accuracy vs. perturbation rate — Homoglyph substitution |
| Fig. 4 | Precision vs. perturbation rate — Synonym substitution |
| Fig. 5 | Precision vs. perturbation rate — Homoglyph substitution |
| Fig. 6 | Heatmap — accuracy and Δ vs. baseline at 30% perturbation, all datasets |
| Fig. 7 | False positive counts — baseline vs. Homoglyph 30% |
| Fig. 8 | False positive counts — baseline vs. Synonym 30% |

---

## Citation

If you use this code, data splits, or trained models, please cite:

```bibtex
@article{arif2026adversarial,
  title   = {Adversarial Robustness of NLP-Based Phishing Email Classifiers: A Multi-Dataset Empirical Study},
  author  = {Arif, Muktadir},
  journal = {Columbia Junior Science Journal},
  year    = {2026}
}
```

---

## Acknowledgements

Thanks to Dr. Latif Siddiq for supervision and guidance throughout this project.

---

## License

[Choose a license — e.g. MIT for code, and check individual dataset licenses on their respective Hugging Face pages before redistribution.]
