# CS5998 Capstone Project: Explainable Activity Recognition from a Smart Textile Pressure Sensor

**Student:** P.B.W. Nanayakkara (268472T) | **Programme:** Masters in Data Science & AI | **Module:** CS5998 - Capstone Project

Recognising 18 everyday activities from a textile pressure-sensor sleeve (the *Smart-Sleeve*) with a 1D-CNN
on the raw multivariate time series, and testing whether time-series explanation methods can be trusted.

## Status
- [x] Repo set up, dataset downloaded and verified
- [x] Segmentation pipeline (2,520 labelled segments)
- [x] EDA and preprocessing
- [x] Baselines (Logistic Regression, Random Forest), 10-fold and LOSO
- [x] 1D-CNN, 10-fold and LOSO (untuned)
- [x] Explainability: CAM, Integrated Gradients, TSR (re-implementation), faithfulness tests, time ablation
- [ ] Stability check (script written: `src/stability.py`; results to be added)
- [x] Plausibility check (`src/plausibility.py`)
- [ ] Demo (script written: `src/demo.py`; needs `src/train_final.py` to be run)
- [ ] Final report

## Dataset
Smart-Sleeve Dataset, Xu, Wan, Deng, Guo & Cheng (2022). Apache-2.0 licence.
- **Data (used in this project):** https://github.com/xghgithub/Smart-Sleeve-Dataset

14 subjects x 10 rounds x 18 activities = 2,520 labelled trials. Each `data.csv` is a raw frame stream
(50 Hz nominal, 12-bit ADC, 20x10 = 200 pressure values per frame, row-first order); `labels.csv` gives each
activity's start/end frame. The sleeve is taken off and re-worn after every round. The raw data is ~520 MB and is
**not** committed; it is downloaded by script.

## Reference
Xu, Wan, Deng, Guo & Cheng (2022). Smart-Sleeve: A Wearable Textile Pressure Sensor Array for Human Activity
Recognition. *Sensors*, 22(5), 1702. https://doi.org/10.3390/s22051702

## Quick start
```bash
pip install -r requirements.txt
python src/download_data.py        # clones the dataset into data/raw/
python src/segment.py              # builds data/processed/{X,y,subject,round,lengths}.npy
python -m src.train_final          # trains and saves the final CNN (models/cnn_final.keras)
python -m src.stability            # stability of explanations across 5 independently trained CNNs
python -m src.plausibility         # needs results/cnn_loso_preds.npy from the notebook's LOSO cell
```
Interactive demo (Colab or Jupyter): `from src.demo import launch; launch()`

## Results so far (random seed 42 unless stated; hardware: Google Colab T4 GPU)
| Model | 10-fold accuracy | LOSO accuracy |
|---|---|---|
| Logistic Regression (summary features) | 68.6% | 59.3% |
| Random Forest (summary features) | 74.3% | 66.5% |
| 1D-CNN (raw sequences) | 74.4% | 64.2% |
| *Published SVM, static features (Xu et al., 2022)* | *82.0%* | *72.7%* |

The published figures use different, hand-made features, so they are a reference and not a like-for-like comparison.

## Explainability findings so far (single dataset, 14 subjects; see the report for limits)
- **Time ablation:** shuffling the frames or replacing every frame with the trial's average frame leaves CNN accuracy unchanged (10-fold means 73.5% / 73.6% / 73.6% for original / shuffled / average-frame input; `results/figures/time_ablation.csv`). The model recognises activities from *where* the sleeve is pressed, not *when*.
- **Spatial faithfulness:** Integrated Gradients and TSR pick sensors that matter far more than random sensors, but rank sensors almost the same way as raw pressure (Spearman 0.99 for IG vs pressure). Single-run AUC gaps between IG/TSR and a highest-pressure baseline are small and change from run to run; the 5-split repeat (`xai_faithfulness_multiseed.csv`) is the more reliable evidence.
- **Temporal faithfulness:** frames chosen by CAM or TSR time relevance are not more important than random frames.
- **Plausibility:** the CNN's confusions follow pressure-pattern similarity (Spearman rho = 0.49 over the 153 activity pairs; pairs are not independent, so the p-value is optimistic). Most-confused pairs include near-duplicates such as writing hunched vs straight (14/15) and the seated variants (16-18).
- TSR is my re-implementation following Ismail et al. (2020) and the TSInterpret reference code (BSD-3), because the library cannot be installed on Python 3.13.

## Repository layout
```
src/            data scripts, shared helpers (common.py), stability, plausibility, final model, demo
notebooks/      EDA and experiments (Colab)
models/         saved final CNN
data/raw/       downloaded dataset (git-ignored)
data/processed/ generated arrays (git-ignored)
results/        metrics (CSV) and figures
report/         milestone documents
```

## Reproducibility
GPU training is not bit-for-bit repeatable: re-running a notebook cell changes accuracies and AUCs by roughly a point (or about 0.01-0.02 AUC), so each result file is the output of one specific run.
Python 3.13 (Colab), TensorFlow 2.21, scikit-learn, SciPy; versions in `requirements.txt`. Seeds: 42 for the main
experiments, seeds 1-5 for the repeated-split experiments. Input: 208 frames x 200 sensors, pressure divided by 4095;
trials longer than 208 frames (123 of 2,520) are truncated, shorter ones zero-padded. CNN: 3 x Conv1D (64, 128, 128;
kernel 5, 5, 3) + BatchNorm + ReLU, global average pooling, dropout 0.3, dense softmax (158,034 parameters). Adam
(lr 0.001), batch 32, up to 60 epochs, early stopping (patience 8) on a validation split taken from the training data only.
