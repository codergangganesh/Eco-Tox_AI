# Dataset Sources for Larger EcotoxAI Training

The current production workflow uses the UCI QSAR Fish Toxicity dataset:
908 compounds, six molecular descriptors, and 96-hour fathead minnow LC50 in
`-LOG(mol/L)`. The model, scaler, prediction API, and web UI all expect these
same six descriptors, so larger datasets should be added only after they are
converted into the same schema:

`CIC0, SM1_Dz_Z, GATS1i, NdsCH, NdssC, MLOGP, LC50`

## Best candidate

1. Zenodo: "Consensus QSAR models estimating acute aquatic toxicity for three
   trophic levels organisms: Algae, Daphnia and Fish"
   - URL: https://zenodo.org/records/3708082
   - DOI: `10.5281/zenodo.3708082`
   - License: CC BY 4.0
   - Why it is relevant: public acute aquatic toxicity collection with fish,
     daphnia, and algae endpoints. The Zenodo record reports 3,680 unique
     compounds and includes fish 96-hour `pLC50` and `LC50` fields plus
     canonical SMILES.
   - Integration note: this is the strongest larger public source found, but it
     is SMILES/SDF based. It should be converted to the six descriptors above
     before being used by the existing training pipeline.

## Additional high-quality sources

2. EPA ECOTOX Knowledgebase
   - URL: https://www.epa.gov/comptox-tools/ecotoxicology-ecotox-knowledgebase-resource-hub
   - Why it is relevant: EPA describes ECOTOX as a curated public knowledgebase
     with over one million test records, more than 13,000 species, and more than
     12,000 chemicals. It is useful for QSAR model building and ecological risk
     assessment.
   - Integration note: excellent long-term source, but records need careful
     filtering by organism, endpoint, exposure duration, units, and chemical
     identity. EPA CTX API access may require an API key.

3. Milano Chemometrics original fish toxicity dataset
   - URL: https://michem.unimib.it/download/data/acute-aquatic-toxicity-to-fish/
   - Why it is relevant: original source for this project's UCI dataset and may
     include molecular SMILES alongside the same descriptors.
   - Integration note: not larger than the current training set, but useful for
     validating chemical identities and deduplicating augmented data.

## How to train from dataset links

Preferred link-based workflow:

1. Add direct CSV or ZIP links to `data/dataset_links.json`.
2. Set `enabled` to `true` only when the source already has this exact schema:
   `CIC0, SM1_Dz_Z, GATS1i, NdsCH, NdssC, MLOGP, LC50`.
3. Run `python train.py`.

The training loader downloads enabled compatible links, caches them in
`data/external/`, appends them to the base dataset, deduplicates exact duplicate
rows, validates numeric values, and keeps all downstream training, prediction,
and web workflows unchanged.

Manual fallback:

Place any curated, same-schema CSV files in `data/external/`.

Each file can either:

- include headers exactly matching `CIC0,SM1_Dz_Z,GATS1i,NdsCH,NdssC,MLOGP,LC50`, or
- use the existing semicolon-delimited, no-header order.

The loader uses the same validation path for link-downloaded and manually placed
files.
