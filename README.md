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

25 of 30 homoglyph/synonym attack conditions were statistically significant after Benjamini-Hochberg FDR correction (bootstrap resampling, n = 1,000); all homoglyph conditions on the three email datasets were significant at every rate tested (p-corrected < 0.001).

---

## Repository Structure

```
.
├── Code/
│   └── multi_dataset_experiment.py   # Single end-to-end pipeline: data loading,
│                                      # DistilBERT fine-tuning, both attacks,
│                                      # and bootstrap/BH-FDR statistics, per dataset
├── results/
│   ├── attack_results_*.csv          # Per-dataset attack results (5 files)
│   ├── stats_*.csv                   # Per-dataset statistical test results (5 files)
│   ├── master_attack_results.csv     # Combined results, all datasets
│   ├── master_statistical_results.csv
│   └── paper_summary_table.csv
├── Figures/
│   ├── fig1_baseline_summary.png
│   ├── fig2_accuracy_combined.png
│   ├── fig3_precision_combined.png
│   ├── fig4_confusion_matrices.png
│   └── fig5_heatmap_summary.png
├── LICENSE
└── README.md
```

> Note: the pipeline is a single consolidated script rather than separate modules — it was developed and run cell-by-cell in Google Colab, then exported as one `.py` file. It includes resume-safe checkpointing (`SKIP_COMPLETED` flag) so it can pick back up after a Colab disconnect without re-running completed datasets.

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

Fine-tuned `distilbert-base-uncased` checkpoints (one per dataset) are hosted on the Hugging Face Hub. All five are grouped in one collection:

**[umayer16/trained-phishing-detection-models](https://huggingface.co/collections/umayer16/trained-phishing-detection-models)**

| Dataset | Model Repo |
|---|---|
| Enron Spam | [`umayer16/distilbert-phishing-enron-spam`](https://huggingface.co/umayer16/distilbert-phishing-enron-spam) |
| Spam Detection | [`umayer16/distilbert-phishing-spam-detection`](https://huggingface.co/umayer16/distilbert-phishing-spam-detection) |
| SMS Spam | [`umayer16/distilbert-phishing-sms-spam`](https://huggingface.co/umayer16/distilbert-phishing-sms-spam) |
| Phishing Email Dataset | [`umayer16/distilbert-phishing-email-dataset`](https://huggingface.co/umayer16/distilbert-phishing-email-dataset) |
| PhishingEmail Detection | [`umayer16/distilbert-phishing-detection-v2`](https://huggingface.co/umayer16/distilbert-phishing-detection-v2) |

Load any model directly with `transformers`:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "umayer16/distilbert-phishing-enron-spam"
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
pip install scikit-learn pandas matplotlib seaborn nltk scipy statsmodels huggingface_hub

# 2. Run the full pipeline (loads all 5 datasets, fine-tunes DistilBERT per
#    dataset, runs both attacks at all 3 rates, computes bootstrap/BH-FDR
#    statistics, and generates figures)
python Code/multi_dataset_experiment.py
```

This script was developed and run on Google Colab (T4 GPU) with Google Drive as persistent storage; the Drive-mounting (`google.colab.drive`) and Hugging Face Hub upload logic near the top of the file are Colab-specific and will need to be adapted (or removed) for a local or other cloud environment. `SKIP_COMPLETED = True` lets the script resume cleanly after an interruption by checking for existing per-dataset result CSVs and model checkpoints before re-running them.

> **Note:** the figure-generation cells in this script currently produce an earlier 8-figure layout (separate synonym/homoglyph panels, separate false-positive bar charts) rather than the 5 consolidated figures shown in `Figures/` and used in the paper. The figure-generation code has not yet been updated to match; running the script end-to-end will not currently reproduce `Figures/` exactly.

---

## Figures

The paper uses five figures:

| Figure | Description |
|---|---|
| Fig. 1 | Baseline classifier accuracy and precision across all five datasets, prior to adversarial perturbation |
| Fig. 2 | Classifier accuracy vs. perturbation rate under (a) synonym-substitution and (b) homoglyph-substitution attacks, all five datasets |
| Fig. 3 | Classifier precision vs. perturbation rate under (a) synonym-substitution and (b) homoglyph-substitution attacks, all five datasets |
| Fig. 4 | Confusion matrices for two representative datasets (Enron Spam, PhishingEmailDetection) under baseline, 30% homoglyph, and 30% synonym conditions |
| Fig. 5 | Accuracy (%) and change from baseline (Δ, percentage points) at 30% perturbation, all five datasets |

---

## Citation

If you use this code, data splits, or trained models, please cite:

```bibtex
@article{arif2026adversarial,
  title   = {Adversarial Robustness of NLP-Based Phishing Email Classifiers: A Multi-Dataset Empirical Study},
  author  = {Arif, Muktadir},
  journal = {XXXXXXXXX(pending)},
  year    = {2026}
}
```

---

## Acknowledgements

Thanks to Dr. Latif Siddiq for supervision and guidance throughout this project.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
