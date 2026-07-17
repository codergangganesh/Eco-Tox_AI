"""
Molecular descriptor generation.

RDKit is optional: the existing six-descriptor QSAR workflow still works when
RDKit is not installed. When a dataset includes SMILES strings, these helpers
add structure-derived descriptors that usually improve QSAR model quality.
"""

import math
from typing import Dict


RDKIT_DESCRIPTOR_NAMES = [
    'MolWt',
    'ExactMolWt',
    'MolLogP',
    'TPSA',
    'NumHDonors',
    'NumHAcceptors',
    'NumRotatableBonds',
    'RingCount',
    'NumAromaticRings',
    'FractionCSP3',
    'HeavyAtomCount',
    'NHOHCount',
    'NOCount',
    'MolMR',
    'BalabanJ',
    'BertzCT',
]


def is_rdkit_available() -> bool:
    try:
        import rdkit  # noqa: F401
        return True
    except ImportError:
        return False


def require_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
        return Chem, Crippen, Descriptors, Lipinski, rdMolDescriptors
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for SMILES descriptor generation. "
            "Install it with: pip install rdkit"
        ) from exc


def calculate_rdkit_descriptors(smiles: str) -> Dict[str, float]:
    """Calculate a compact, stable set of RDKit descriptors from one SMILES."""
    Chem, Crippen, Descriptors, Lipinski, rdMolDescriptors = require_rdkit()

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    values = {
        'MolWt': Descriptors.MolWt(mol),
        'ExactMolWt': Descriptors.ExactMolWt(mol),
        'MolLogP': Crippen.MolLogP(mol),
        'TPSA': rdMolDescriptors.CalcTPSA(mol),
        'NumHDonors': Lipinski.NumHDonors(mol),
        'NumHAcceptors': Lipinski.NumHAcceptors(mol),
        'NumRotatableBonds': Lipinski.NumRotatableBonds(mol),
        'RingCount': rdMolDescriptors.CalcNumRings(mol),
        'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),
        'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
        'NHOHCount': Lipinski.NHOHCount(mol),
        'NOCount': Lipinski.NOCount(mol),
        'MolMR': Crippen.MolMR(mol),
        'BalabanJ': Descriptors.BalabanJ(mol),
        'BertzCT': Descriptors.BertzCT(mol),
    }

    clean_values = {}
    for name in RDKIT_DESCRIPTOR_NAMES:
        value = float(values[name])
        if not math.isfinite(value):
            raise ValueError(f"RDKit descriptor {name} is not finite for SMILES: {smiles}")
        clean_values[name] = value
    return clean_values

