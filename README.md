# CS5998 Capstone Project: Explainable Activity Recognition from a Smart Textile Pressure Sensor

**Student:** P.B.W. Nanayakkara (268472T) | **Programme:** Masters in Data Science & AI | **Module:** CS5998 - Capstone Project

Recognising 18 everyday activities from a textile pressure-sensor sleeve (the *Smart-Sleeve*). Three classical models on
summary features are compared with a 1D-CNN on the raw multivariate time series, under 10-fold cross-validation and
leave-one-subject-out (LOSO) evaluation with 95% subject-level confidence intervals. The CNN is then used to test whether
time-series explanation methods (CAM, Integrated Gradients, Temporal Saliency Rescaling) can be trusted.

**Demo video:** *(add the YouTube link here)*

## Headline findings
1. **Best model depends on the protocol.** An SVM is best under 10-fold (77.2%). For a *new person* (LOSO), Random Forest, SVM and the CNN (64.7-66.5%) cannot be told apart statistically; Random Forest and SVM beat Logistic Regression by about 7 points and the CNN by about 5. All are below the published SVM (82.0% / 72.7%), which used different hand-made features.
2. **The CNN recognises activities from where the sleeve is pressed, not when.** Shuffling the frames or replacing every frame with the trial's average frame leaves accuracy unchanged.
3. **Explanations are faithful but mostly follow raw pressure.** Integrated Gradients and TSR pick sensors that matter far more than random ones, but rank sensors almost exactly like raw pressure (Spearman 0.99 / 0.98).
4. **Generalising to a new person is the main limit.** Per-subject LOSO accuracy ranges from 48.9% to 86.7% and is not explained by arm length, forearm length or biceps circumference.

## Status
- [x] Repo set up, dataset downloaded and verified
- [x] Segmentation pipeline (2,520 labelled trials) and data-quality checks
- [x] EDA and preprocessing
- [x] Baselines (Logistic Regression, Random Forest, SVM) and 1D-CNN, 10-fold and LOSO, with confidence intervals
- [x] Error analysis and per-subject / body-measurement check
- [x] Explainability: time ablation, CAM, Integrated Gradients, TSR (re-implementation), deletion/insertion and temporal faithfulness
- [x] Robustness: five-split faithfulness and stability across trained models
- [x] Plausibility check and interactive demo
- [ ] Demo video link
- [ ] Final report

## Dataset
Smart-Sleeve Dataset, Xu, Wan, Deng, Guo & Cheng (2022). Apache-2.0 licence.
- **Data (used in this project):** https://github.com/xghgithub/Smart-Sleeve-Dataset

14 subjects x 10 rounds x 18 activities = 2,520 labelled trials. Each `data.csv` is a raw frame stream (about 52.6 Hz
measured, 12-bit ADC, 20x10 = 200 pressure values per frame, row-first order); `labels.csv` gives each activity's start
and end frame. The sleeve is taken off and re-worn after every round. The raw data is about 520 MB and is **not**
committed; it is downloaded by script.

## Reference
Xu, Wan, Deng, Guo & Cheng (2022). Smart-Sleeve: A Wearable Textile Pressure Sensor Array for Human Activity
Recognition. *Sensors*, 22(5), 1702. https://doi.org/10.3390/s22051702

## How to reproduce
Open the notebooks in `notebooks/final/` in Google Colab and run them **in order**. Use a T4 GPU for notebooks 02-05
(Runtime > Change runtime type). Each notebook starts with the same setup cell, ends with a self-check
("ALL PASS") and downloads its results. All result files (CSV, NPY and PNG) are stored in `results/figures/`.

| Notebook | Purpose | Main outputs |
|---|---|---|
| `01_data_pipeline_and_eda` | download, segmentation, data-quality checks, EDA | `data_quality_summary.csv`, `eda_*.png`, `segment_examples.png` |
| `02_models_and_evaluation` | LR, RF, SVM, 1D-CNN (10-fold and LOSO), confidence intervals, error analysis, body-size check | `baselines.csv`, `cnn_*`, `model_comparison_ci.csv`, `paired_differences_ci.csv`, `top_confused_pairs.csv` |
| `03_explainability_experiments` | time ablation, CAM, Integrated Gradients, TSR, deletion/insertion and temporal faithfulness | `time_ablation.csv`, `xai_*.csv`, `xai_*.png` |
| `04_robustness_five_split_and_stability` | five-split faithfulness, stability across five trained models | `xai_faithfulness_multiseed.csv`, `xai_stability.csv` |
| `05_final_model_plausibility_and_demo` | final model, plausibility check, interactive demo, pinned requirements | `models/cnn_final.keras`, `plausibility_*` |

Scripts used by the notebooks live in `src/` (run from the repository root, e.g. `python -m src.stability`).
Interactive demo: `from src.demo import launch; launch()`.

## Results
### Model comparison (accuracy, 95% subject-level bootstrap interval in brackets)
| Model | 10-fold | LOSO (new person) |
|---|---|---|
| Logistic Regression (baseline, summary features) | 68.6% [64.1, 73.1] | 59.3% [53.6, 65.0] |
| Random Forest (summary features) | 74.3% [68.9, 79.7] | 66.5% [59.8, 73.2] |
| SVM, RBF, C tuned (summary features) | **77.2%** [71.4, 82.8] | 66.4% [59.8, 72.9] |
| 1D-CNN (raw sequences) | 73.7% [68.9, 78.2] | 64.7% [59.4, 70.4] |
| *Published SVM, different features (Xu et al., 2022)* | *82.0%* | *72.7%* |

The 14 subjects are resampled with replacement (5,000 resamples) because trials of one person are not independent.
Paired differences (`paired_differences_ci.csv`): under 10-fold the SVM beats Random Forest by 2.9 points (95% CI 0.8 to
5.0) and the CNN by 3.6 (1.4 to 5.6), while the CNN and Random Forest are not clearly different (-0.7, CI -2.7 to 1.3).
Under LOSO, no difference among Random Forest, SVM and CNN is clear (all intervals include 0); Random Forest and SVM beat Logistic
Regression by about 7 points (intervals exclude 0); the CNN is about 5 points above it (not part of the paired table).

### Data quality (`data_quality_summary.csv`)
- 140 round files, 2,520 trials, exactly 140 per activity and 180 per subject.
- 90 of 140 files contain one truncated row (a dropped packet); **none** lies inside a labelled trial.
- Trial length: median 124 frames, 95th percentile 208, maximum 358. Trials are padded or truncated to 208 frames (123 trials truncated).

### Explainability (single trained CNN unless stated)
- **Time ablation** (`time_ablation.csv`): 10-fold accuracy 73.7% (original), 73.5% (frames shuffled), 73.6% (every frame replaced by the trial average).
- **Spatial faithfulness:** deletion AUC (lower is more faithful) IG 0.244, highest-pressure baseline 0.253, random 0.649; insertion AUC (higher is better) 0.281, 0.281, 0.076. Rank correlation of sensor importance: IG vs pressure 0.99, TSR vs pressure 0.98.
- **Five random splits** (`xai_faithfulness_multiseed.csv`): deletion AUC IG 0.253 vs pressure 0.264 vs random 0.664 (gap 0.011, SD 0.005, IG ahead in 5 of 5 splits); insertion AUC 0.334 vs 0.331 vs 0.077 (gap 0.003, SD 0.013, 3 of 5). IG adds only a small amount beyond raw pressure.
- **TSR** (`xai_tsr_faithfulness.csv`): deletion AUC TSR 0.277, IG 0.279, pressure 0.272, random 0.628: no better than the pressure baseline.
- **Temporal faithfulness** (`xai_temporal_faithfulness_*.csv`): replacing 50-70% of frames by the trial average leaves accuracy at 70.1-70.6% (70.2% at baseline). Blanking frames to zero: CAM and TSR-selected frames are not consistently more important than random frames (at 50% blanked: CAM 55.0%, TSR 57.1%, random 53.8%; at 70%: 23.0%, 27.6%, 33.6%).
- **Stability** (`xai_stability.csv`, five CNNs differing only in random seed, 200 trials): IG sensor rankings agree across models (Spearman 0.99, top-10% sensor overlap 0.66) but also agree with raw pressure (0.995); a different trial of the same activity agrees far less (0.24), so the maps are trial-specific. CAM time curves correlate 0.70 across models.
- **Plausibility** (`plausibility_summary.txt`): the CNN confuses activities whose mean pressure patterns look alike (Spearman 0.49 over the 153 activity pairs; pairs are not independent, so the p-value is optimistic). Most confused: Hold cheeks / Play mobile phone (22%), Lean forward at work / Sit with hands on the armrests, Fold arms / Hug a doll, Write with a hunchback / straight back.
- **Per subject** (`subject_accuracy_vs_anthropometrics.csv`): CNN LOSO accuracy ranges from 48.9% (subject 12) to 86.7% (subject 10). Rank correlations with arm length, forearm length and biceps circumference are weak (|rho| at most 0.40, all p > 0.15; n = 14, exploratory).

## Limitations
- Only 14 subjects and one dataset: the confidence intervals are wide (about +/-5 points), which is why most model differences under LOSO are not clear.
- The CNN is untuned, and it sees only about 11 consecutive frames at a time (three convolutions, kernels 5, 5 and 3), so the finding that time order does not matter applies to this model and these mostly stationary activities, not to temporal models in general.
- GPU training is not bit-for-bit repeatable: re-running a notebook changes accuracies by roughly a point and AUCs by 0.01-0.02, so each result file is the output of one specific run.
- Deletion and insertion tests feed the CNN inputs it never saw in training; compare methods with each other and do not read the absolute accuracies.
- TSR is my re-implementation following Ismail et al. (2020) and the TSInterpret reference code (BSD-3), because the library cannot be installed on Colab's Python 3.13.
- Plausibility uses pattern similarity, not anatomical ground truth, because the dataset does not document which end of the sleeve is the wrist.
- The published SVM's hand-made features were not reproduced, so the comparison with the paper is not like for like.

## Repository layout
```
README.md, requirements.txt, .gitignore
notebooks/final/  the five notebooks that reproduce every reported result (run 01 to 05 in order)
src/              data scripts, shared helpers (common.py), stability, plausibility, final model, demo
models/           saved final CNN (cnn_final.keras) and its hold-out indices
results/figures/  all result files: CSV, NPY and PNG, plus plausibility_summary.txt
data/raw/         downloaded dataset (git-ignored)
data/processed/   generated arrays (git-ignored)
```

## Reproducibility checklist (Appendix A)
- **Dependencies:** `requirements.txt` (versions pinned from the Colab environment of the final run).
- **Hardware:** Google Colab, T4 GPU (notebooks 02-05); Python 3.13, TensorFlow 2.21.
- **Random seeds:** 42 for the main experiments and final model; seeds 1-5 for the five-split and stability experiments.
- **Data availability:** public, https://github.com/xghgithub/Smart-Sleeve-Dataset (`src/download_data.py`).
- **Splits:** 10-fold CV is stratified and shuffled (seed 42); LOSO uses 14 folds (one subject held out each time); in every neural-network fold 10% of the training part (stratified) is held out for validation and early stopping; the explainability experiments use a stratified 80/20 split (the seed sets the split); the final model uses a stratified 72/8/20 train/validation/test split (seed 42).
- **Preprocessing:** pressure divided by 4095; trials padded with zeros or truncated to 208 frames (95th percentile of trial length); summary features for the classical models are the mean, standard deviation and maximum of each of the 200 sensors over a trial's true frames (600 features).
- **Classical models:** Logistic Regression (standardised, max_iter 2000); Random Forest (200 trees, seed 42); SVM (RBF, standardised, `C` in {1, 10, 100} chosen by 3-fold CV inside each training set).
- **1D-CNN:** 3 x Conv1D (64, 128, 128 filters; kernels 5, 5, 3) each with BatchNorm and ReLU, global average pooling, dropout 0.3, dense softmax over 18 classes (158,034 parameters); Adam (learning rate 0.001), batch size 32, up to 60 epochs, early stopping on validation loss (patience 8, best weights restored).
