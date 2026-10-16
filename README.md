# CS5998 Capstone: Explainable Activity Recognition from a Smart Textile Pressure Sensor

**Student:** P.B.W. Nanayakkara (268472T) | **Programme:** MSc Data Science & AI | **Module:** CS5998

Recognising 18 everyday activities from a textile pressure-sensor sleeve (the *Smart-Sleeve*),
treating the data as a multivariate time series, and explaining predictions with a
time-series-specific explainability method.

## Status
- [x] Repo set up, dataset downloaded and verified
- [x] Segmentation pipeline (2,520 labelled segments)
- [ ] EDA and preprocessing
- [ ] Baselines (Logistic Regression, Random Forest)
- [ ] 1D-CNN
- [ ] Time-series explainability
- [ ] Final report

## Dataset
Smart-Sleeve Dataset, Xu, Wan, Deng, Guo & Cheng (2022). Apache-2.0 licence.
- Data repository: https://github.com/xghgithub/Smart-Sleeve-Dataset
- Paper (open access): https://www.mdpi.com/1424-8220/22/5/1702 (DOI 10.3390/s22051702)
- Paper (PMC mirror): https://pmc.ncbi.nlm.nih.gov/articles/PMC8914988

14 subjects x 10 rounds x 18 activities = 2,520 labelled trials. Each `data.csv` is a raw
frame stream (~52 Hz, 20x10 = 200 pressure values per frame, range 0-4095); `labels.csv`
gives each activity's start/end frame. The raw data is ~520 MB and is **not** committed;
it is downloaded by script.

## Quick start
```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/download_data.py     # clones the dataset into data/raw/
python src/segment.py           # builds data/processed/{X,y,subject,round,lengths}.npy
python src/sanity_check.py      # plots a few segments to confirm alignment
```

## Repository layout
```
src/            download_data.py, segment.py, sanity_check.py (+ models later)
notebooks/      EDA and experiments
data/raw/       downloaded dataset (git-ignored)
data/processed/ generated arrays (git-ignored)
results/        metrics and figures
report/         milestone documents
```

## Reproducibility
Random seed 42 everywhere; Python and package versions in `requirements.txt`.
Hardware, splits and hyperparameters will be recorded here as the project progresses.
