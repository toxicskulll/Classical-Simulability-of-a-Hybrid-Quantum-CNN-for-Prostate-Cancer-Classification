# Data

The input data are not distributed with this repository. The saved results in `../results/` do not need them;
the files below are needed only to rerun training or the data checks.

## PROSTATEx lesion table: `prostatex_lesions.csv`

One row per PROSTATEx training finding, 360 columns (the full list, in order, is in `columns.txt`):

| Column | Meaning |
|---|---|
| `ProxID`, `fid` | PROSTATEx patient identifier and finding number; together they identify a lesion |
| `label` | 1 if the finding is clinically significant prostate cancer (`ClinSig` in the PROSTATEx findings file), else 0 |
| `clin_psa`, `clin_psad`, `clin_prostate_volume`, `clin_patient_age` | pre-biopsy clinical variables: PSA, PSA density, prostate volume, age |
| `leak_gleason_grade` | Gleason grade; used only by the leakage-reference model and refused as an input by every other model |
| `t2w_*`, `adc_*`, `dwi_*`, `ktrans_*` | 352 PyRadiomics features, 88 per image type |

The table used in the paper has 327 lesions from 204 patients, 76 of them clinically significant. Its SHA-256
checksum is

```text
cc3ca3532c257e9f00974167b2eefe4ca13ad6223768a8c79fc85fdf473f54df
```

It was built from the PROSTATEx training data (The Cancer Imaging Archive, doi:10.7937/K9TCIA.2017.MURS5CL) as
follows:

- radiomic features were extracted with PyRadiomics from the T2-weighted, ADC, DWI and K<sup>trans</sup> images at
  each finding;
- labels were derived from `ClinSig` in `ProstateX-Findings-Train.csv`;
- three `(ProxID, fid)` keys are shared by two findings whose extracted features are identical, so one row of each
  pair was removed; the one pair that combines a clinically significant and a non-significant finding
  (`ProstateX-0005`, finding 1) is labelled positive (`duplicate_label_policy: max` in `configs/champion.yaml`).

`manifest.json` records the removed rows, the shared keys and the checksum of the findings file.

`v3core/data.py` checks the counts at load time and refuses any model input whose name contains `gleason`, `isup`,
`clinsig`, `cspca`, `grade_group`, `label` or `leak_`. `scripts/check_data.py` runs the full set of data checks.

## PI-CAI (external evaluation only)

`scripts/picai_transfer.py` expects

- `picai/features_cache.csv`: one row per PI-CAI case, with `ProxID`, `fid`, the clinical columns
  `clinical_psa`, `clinical_psad`, `clinical_prostate_volume`, `clinical_patient_age`, and the `t2w_*`, `adc_*` and
  `dwi_*` radiomic columns of the PROSTATEx table (PI-CAI has no K<sup>trans</sup>);
- `picai/labels_cache.csv`: `ProxID`, `fid`, `label` (case-level csPCa).

`scripts/picai_overlap.py` expects `ProstateX-mapping.json`, the list of PI-CAI cases that come from PROSTATEx,
distributed with the PI-CAI labels. The PI-CAI data are available from the PI-CAI challenge
(doi:10.5281/zenodo.6624726).
