# Data Directory

Place the UCI QSAR Fish Toxicity dataset here.

## Download Instructions

1. Go to https://archive.ics.uci.edu/dataset/504/qsar+fish+toxicity
2. Click **Download**
3. Extract and place the CSV file as `qsar_fish_toxicity.csv` in this folder

**OR** — the training script will download it automatically using the `ucimlrepo` package.

## Dataset Details

- **908 chemicals** with experimentally measured LC50 values
- **6 molecular descriptors**: CIC0, SM1_Dz(Z), GATS1i, NdsCH, NdssC, MLOGP
- **Target**: LC50 [-LOG(mol/L)] for fathead minnow (Pimephales promelas), 96-hour exposure
