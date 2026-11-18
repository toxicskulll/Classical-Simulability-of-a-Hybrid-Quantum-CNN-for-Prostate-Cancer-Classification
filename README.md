# Classical Simulability and NISQ Feasibility of a Tree-Structured Hybrid QCNN for Prostate Cancer Classification

Code, configuration and results for the paper of the same title. The repository contains the hybrid quantum
convolutional neural network (HQCNN), its exact classical twin, the classical baselines and ablation arms, every
experiment script, and the saved outputs from which every table and figure of the paper is generated.

## Contents

```text
configs/            experiment configurations (YAML)
  champion.yaml       full specification: data, cross-validation, preprocessing, encoding, circuit, training
  phase2.yaml         mechanism-ablation arms (extends champion.yaml)
  phase3.yaml         nested tuning and tuned models (extends champion.yaml)
v3core/             library
  config.py           configuration loading; paths resolve relative to the repository root
  data.py             lesion table, leakage guard, patient-grouped folds
  preprocess.py       fold-local imputation, scaling, feature selection, PCA, resampling
  hqcnn.py            the HQCNN (PennyLane)
  twin.py             exact classical twin of the HQCNN (linear in the number of qubits)
  arms.py, phase2.py  ablation arms (dephased control, no entanglement, frozen circuits, tree networks, ...)
  models.py           classical baselines (logistic regression, SVM, random forest, MLP, clinical models)
  noise.py            gate and shot noise via the twin
  tuning.py           nested hyperparameter search (Optuna)
  cv.py, seeding.py   cross-validation runner and per-component seeding
  stats.py, metrics.py  AUC, patient-level bootstrap, paired DeLong test, threshold metrics
  plotting.py         figure style (vector PDF, text of at least 8 pt)
scripts/            entry points (see "Reproducing the paper")
data/               expected input files (not distributed; see data/README.md)
results/            saved outputs of every experiment (see "Results")
docs/simulability.md  derivation of the classical twin and its limits
```

## Installation

The experiments were run with Python 3.9 on Windows 11 (AMD Ryzen 7 6800H, 8 cores, 15 GB RAM). Either

```bash
conda env create -f environment.yml
conda activate hqcnn-prostate
```

or

```bash
python -m pip install -r requirements.txt
```

All commands below are run from the repository root. Set `PYTHONWARNINGS=ignore` to hide library deprecation
warnings.

## Data

The lesion table (`data/prostatex_lesions.csv`) is not distributed with this repository. The PROSTATEx data are
available from The Cancer Imaging Archive. `data/README.md` describes the expected file, its columns and how it was
built, and gives the SHA-256 checksum of the table used in the paper. The saved results in `results/` do not need
the data: every table and figure can be regenerated from them directly.

## Quick checks

These run in seconds to minutes and need no training:

```bash
python scripts/check_data.py            # counts, patient-disjoint folds, leakage guard, training-fold-only fitting
python scripts/check_hqcnn.py           # simulator agreement and determinism of the HQCNN
python scripts/twin_equivalence.py      # classical twin versus statevector simulation on random inputs
python scripts/verify_bilinear.py       # each tree node is bilinear in the Bloch coordinates
python scripts/check_phase2_arms.py     # correctness of the ablation arms
```

`check_data.py`, `check_hqcnn.py` and `check_phase2_arms.py` need the lesion table.

## Reproducing the paper

Each (model, seed, fold) task is independent, runs in parallel with `--n-jobs`, and is skipped if its output
already exists, so interrupted runs resume. Seeds 0 to 4 each give a different patient-grouped five-fold partition.
Wall times are for the machine above with the stated number of workers.

The code comments and folder names refer to three phases of the work: Phase 1 is the leakage-free baseline, Phase 2
the classical twin and the mechanism ablations, and Phase 3 the tuning, noise, routing and external-data analyses
(`configs/phase2.yaml`, `results/phase3/`, and so on).

### Data and leakage-free baseline

```bash
python scripts/run_cv.py --n-jobs 5          # 7 models x 5 seeds x 5 folds; about 27 min
python scripts/summarise.py                  # results/tables/
```

### Model

```bash
python scripts/circuit_figures.py            # pipeline and circuit diagrams, results/figures/
python scripts/param_counts.py               # results/param_counts.csv
```

### Classical simulability

```bash
python scripts/twin_equivalence.py                 # twin versus statevector, several configurations
python scripts/twin_trained_check.py --n-jobs 5    # the twin reproduces every trained out-of-fold prediction
python scripts/twin_scaling.py                     # forward-pass time versus qubit count (run on an idle machine)
python scripts/verify_bilinear.py
```

### Mechanism ablations

```bash
python scripts/run_cv.py --config configs/phase2.yaml --n-jobs 5   # about 70 min
python scripts/summarise.py --config configs/phase2.yaml           # results/tables_phase2/
python scripts/brickwork_analysis.py --n-jobs 3                    # tree versus brickwork, results/phase2/brickwork/
```

### Scalability and trainability

```bash
python scripts/gradvar.py --run pca_n --n-jobs 4        # gradient variance versus qubit count
python scripts/gradvar.py --run champion12 --n-jobs 4
python scripts/gradvar.py --run focal8 --n-jobs 2
python scripts/qubit20_check.py                         # cost of a 20-qubit model
```

The entanglement entropies are computed by `brickwork_analysis.py` above.

### NISQ feasibility

```bash
python scripts/noise_study.py --stages verify train6q eval summary --n-jobs 5   # gate and shot noise
python scripts/noise_study.py --stages trainnoisy summary --n-jobs 5            # noise-aware training; about 86 min
python scripts/phase3_extras.py                                                  # paired noise comparisons
python scripts/linear_routing.py                                                 # SWAP and CNOT counts
```

The `eval` stage also evaluates the tuned weights, so run it after the nested tuning below.

### Clinical benchmark

```bash
python scripts/tune_nested.py --verify                   # twin HQCNN matches PennyLane for one configuration
python scripts/tune_nested.py --n-jobs 7                 # 7 models x 5 seeds x 5 outer folds; about 11 h
python scripts/run_cv.py --config configs/phase3.yaml --n-jobs 7 \
    --models hqcnn_tuned dephased_tuned tree_net_tuned mlp_matched_tuned lr_tuned svm_tuned rf_tuned
python scripts/summarise.py --config configs/phase3.yaml # results/tables_phase3/
python scripts/compare_vs_clinical.py                    # tuned models versus the clinical logistic regression
```

The external PI-CAI evaluation needs PI-CAI radiomic features that are not distributed (`data/README.md`).
`picai_transfer.py` trains on PROSTATEx and writes the predictions to `results/phase3/picai/`;
`picai_overlap.py` recomputes every PI-CAI number from those saved predictions, with and without the cases that
PI-CAI shares with PROSTATEx:

```bash
python scripts/picai_transfer.py --n-jobs 5              # needs data/picai/; about 11 min
python scripts/picai_overlap.py                          # needs data/ProstateX-mapping.json
```

### Tables

```bash
python scripts/make_paper_tables.py                      # every table of the paper, results/paper_tables/
```

The script reads only the saved CSV and JSON results, so it runs without the data or any training.

## Results

| Folder | Contents |
|---|---|
| `results/oof/<model>/` | out-of-fold probabilities for every seed and fold (`seed<S>_fold<F>.csv.gz`) and run metadata (`.meta.json`: fold metrics, selected features, loss history, a digest of the exact training and test inputs) |
| `results/tables*/` | cross-validation metrics, pooled AUC with patient-bootstrap intervals, paired DeLong tests |
| `results/phase2/` | twin checks, trained twin parameters, brickwork analysis, gradient variance |
| `results/phase3/` | tuning studies, noise study, routing counts, PI-CAI predictions and summaries, 20-qubit check |
| `results/paper_tables/` | the LaTeX tables of the paper (`main_*` main text, `s_*` supplement) |
| `results/figures/` | the figures of the paper (vector PDF) |

## Licence

The code is released under the Apache License 2.0 (`LICENSE`). The PROSTATEx and PI-CAI data are subject to the
terms of their providers.

## Citation

If you use this code, please cite the paper (`CITATION.cff`) and the datasets:

- G. Litjens, O. Debats, J. Barentsz, N. Karssemeijer and H. Huisman, "Computer-Aided Detection of Prostate Cancer in
  MRI", *IEEE Transactions on Medical Imaging*, 33(5):1083–1092, 2014. doi:10.1109/TMI.2014.2303821
- S. G. Armato et al., "PROSTATEx Challenges for computerized classification of prostate lesions from
  multiparametric magnetic resonance images", *Journal of Medical Imaging*, 5(4):044501, 2018.
  doi:10.1117/1.JMI.5.4.044501
- G. Litjens, O. Debats, J. Barentsz, N. Karssemeijer and H. Huisman, "SPIE-AAPM PROSTATEx Challenge Data", The Cancer
  Imaging Archive, 2017. doi:10.7937/K9TCIA.2017.MURS5CL
- K. Clark et al., "The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository",
  *Journal of Digital Imaging*, 26(6):1045–1057, 2013. doi:10.1007/s10278-013-9622-7
- A. Saha et al., "The PI-CAI Challenge: Public Training and Development Dataset", Zenodo, 2022.
  doi:10.5281/zenodo.6624726
