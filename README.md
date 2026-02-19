# Slab Outline Detection

This project focuses on detecting slab outlines using contour-based feature extraction and an SVM classifier.

---

## 📁 Project Structure

```
slab_outline/
│
├── data_/
│ ├── 402627 - BHC Chermside McNabs/
│ │ ├── images/
│ │
│ ├── 402867 - Polycell - The Rochester/
│ ├── 402868 - Melrose Built - Ducale Luxury Residence, Teneriffe/
│ ├── 402886 - M8 Con Trading - Kora, 798 Pacific Pd, Currumbin/
│ ├── Res-7 level-PoC/
│ ├── Resi-6 lvls (402460 - One Earle Lane)/
│ ├── Resi-6 lvls (402683 - Hutchinson Radcliffe Kingsford Tce)/
│ ├── Resi-8 lvls (402489 - Lana Apartments)/
│ ├── Resi-8 lvls (402503-Paynters-Southport)/
│ ├── Residential - 5 level/
│
├── data_utils/
│ ├── dataset_with_features.csv
│ └── model.pkl
│
├── runs/
│ └── SO_DET_V1/
│
├── src/
│ └── slab_outline_detector.py
│
├── .gitignore
└── README.md
```


---

## 📂 Folder Description

- **data_/** → Raw project data 
- **data_utils/** → Preprocessed data CSV with contour features and labels and trained model files  
- **runs/** → Experiment outputs and inference results  
- **src/** → Core source code for slab outline detection
- **.gitignore** → Git ignore rules  
- **README.md** → Project documentation  

---

## 🚀 Workflow Overview

1. Extract contour features from slab images.
2. Normalize multi-kernel contour features.
3. Train SVM classifier.
4. Predict optimal contour.
5. Save visualization and CSV outline results.

---

## 🧠 Model

- Classifier: SVM
- Feature Scaling: StandardScaler
- Input: Relative contour features
- Output: Selected optimal slab contour

---

## 📌 Notes

- Model file stored in `data_utils/model.pkl`
- Feature dataset stored in `data_utils/dataset_with_features.csv`
- Inference outputs stored in `runs/SO_DET_V1/`

---
