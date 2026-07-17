"""
Build data/toxicity_with_smiles.csv from the downloaded empirical aquatic
ecotoxicity TSV.

The output target is LC50 in -LOG(mol/L), matching the existing project.
"""

import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from src.data_loader import DATA_DIR


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_PATH = os.path.join(DATA_DIR, 'external', 'experimental_dataset.tsv')
OUTPUT_PATH = os.path.join(DATA_DIR, 'toxicity_with_smiles.csv')


def smiles_mol_weight(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return float(Descriptors.MolWt(mol))


def build_dataset(source_path: str = SOURCE_PATH, output_path: str = OUTPUT_PATH):
    cols = [
        'original_SMILES',
        'Endpoint',
        'Conc_mgL',
        'Conc_sign',
        'EFSA_species_group',
        'ECOTOX_Species_group',
    ]
    df = pd.read_csv(source_path, sep='\t', usecols=cols, low_memory=False)

    fish_mask = (
        df['EFSA_species_group'].astype(str).str.contains('Fish', case=False, na=False)
        | df['ECOTOX_Species_group'].astype(str).str.contains('Fish', case=False, na=False)
    )
    sign = df['Conc_sign'].fillna('').astype(str).str.strip()
    clean = df[
        df['Endpoint'].astype(str).str.upper().eq('LC50')
        & fish_mask
        & sign.isin(['', '='])
        & df['original_SMILES'].notna()
    ].copy()

    clean['Conc_mgL'] = pd.to_numeric(clean['Conc_mgL'], errors='coerce')
    clean = clean[clean['Conc_mgL'] > 0]

    mw_cache = {}
    rows = []
    bad_smiles = 0
    for smiles, conc_mg_l in zip(clean['original_SMILES'], clean['Conc_mgL']):
        if smiles not in mw_cache:
            mw_cache[smiles] = smiles_mol_weight(smiles)
        mol_weight = mw_cache[smiles]
        if mol_weight is None or mol_weight <= 0:
            bad_smiles += 1
            continue

        molar_lc50 = conc_mg_l / (mol_weight * 1000.0)
        rows.append((smiles, -np.log10(molar_lc50)))

    out = pd.DataFrame(rows, columns=['SMILES', 'LC50'])
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out.groupby('SMILES', as_index=False)['LC50'].median()
    out = out[(out['LC50'] >= 0) & (out['LC50'] <= 12)]
    out.to_csv(output_path, index=False)

    return {
        'source_rows': int(len(df)),
        'usable_fish_lc50_rows': int(len(clean)),
        'bad_smiles': int(bad_smiles),
        'unique_smiles': int(len(out)),
        'output_path': output_path,
        'lc50_min': float(out['LC50'].min()) if len(out) else None,
        'lc50_max': float(out['LC50'].max()) if len(out) else None,
        'lc50_mean': float(out['LC50'].mean()) if len(out) else None,
    }


if __name__ == '__main__':
    summary = build_dataset()
    for key, value in summary.items():
        print(f"{key}: {value}")
