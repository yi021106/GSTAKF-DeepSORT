# GSTAKF-DeepSORT

GSTAKF-DeepSORT is a modular research implementation of the Gaussian Strong Tracking Adaptive Kalman Filter described in *An Improved GSTAKF-Based Pedestrian Tracking Method for Abrupt State Estimation*. The project uses the original DeepSORT tracking-by-detection pipeline as its base and isolates the motion filters, artificial perturbation model, data association, and experiment runner into separate modules.

No dataset, detection file, ReID model, or experiment output is included in this repository.

## Methods

The following motion filters share the same DeepSORT detector input, appearance metric, matching cascade, and track-management logic:

- `standard`: original DeepSORT Kalman filter
- `nsa`: confidence-dependent NSA-only Kalman filter
- `stf`: residual-driven strong tracking filter
- `gstakf`: proposed Gaussian strong tracking adaptive Kalman filter

Each active trajectory owns an independent filter instance and residual history.
For NSA-only, detection confidence is used in both the measurement update and
Mahalanobis gating, matching the benchmark implementation used in the study.

## Structure

```text
GSTAKF-DeepSORT/
├── application_util/
├── configs/
│   └── paper_parameters.json
├── deep_sort/
│   ├── filters/
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── gstakf.py
│   │   ├── nsa.py
│   │   ├── standard.py
│   │   └── stf.py
│   ├── detection.py
│   ├── linear_assignment.py
│   ├── perturbation.py
│   ├── track.py
│   └── tracker.py
├── tools/
├── deep_sort_app.py
├── evaluate_motchallenge.py
└── run_paper_experiments.py
```

## Installation

```bash
git clone https://github.com/yi021106/GSTAKF-DeepSORT.git
cd GSTAKF-DeepSORT
python -m pip install -r requirements.txt
```

Python 3.9 or later is recommended.

## Data

Download MOT17 or MOT20 separately from MOTChallenge. Precomputed DeepSORT detections must use the original `Nx138` NumPy format: the first ten columns contain MOT detection data and the remaining 128 columns contain appearance descriptors.

```text
MOT17/
├── train/
│   ├── MOT17-02-DPM/
│   ├── MOT17-04-DPM/
│   └── ...
└── detections_mot17_train/
    ├── MOT17-02-DPM.npy
    ├── MOT17-04-DPM.npy
    └── ...
```

## Single-Sequence Run

```bash
python deep_sort_app.py \
  --sequence_dir /path/to/MOT17/train/MOT17-02-DPM \
  --detection_file /path/to/detections/MOT17-02-DPM.npy \
  --output_file outputs/MOT17-02-DPM.txt \
  --filter_type gstakf \
  --noise_level high \
  --min_confidence 0.3 \
  --nn_budget 100 \
  --display False
```

Available perturbation levels are:

| Level | Position range | Scale range | Injection probability |
|---|---:|---:|---:|
| `none` | 0 px | 0% | 0 |
| `medium` | ±15 px | ±25% | 0.3 |
| `high` | ±25 px | ±40% | 0.5 |

The perturbation module does not set a fixed random seed.
For an exact diagnostic rerun of the manuscript perturbation sequence, pass
`--noise_seed 2025`. If this option is omitted, the seed remains unfixed.

## Paper Configuration

The default GSTAKF parameters are:

| Parameter | Value |
|---|---:|
| `rho` | 0.82 |
| `beta` | 1.0 |
| `lambda_center` | 0.0 |
| `conf_center` | 0.5 |
| `sigma_lambda` | 0.5 |
| `sigma_conf` | 0.3 |
| `sigmoid_k` | 8.0 |
| `sigmoid_x0` | 0.75 |

The same values are stored in `configs/paper_parameters.json`.

## Batch Experiment

The batch runner generates TrackEval-compatible tracker folders for all requested filter and perturbation combinations:

```bash
python run_paper_experiments.py \
  --mot_dir /path/to/MOT17/train \
  --detection_dir /path/to/detections_mot17_train \
  --output_dir outputs/MOT17-train
```

To run only the high-perturbation comparison:

```bash
python run_paper_experiments.py \
  --mot_dir /path/to/MOT17/train \
  --detection_dir /path/to/detections_mot17_train \
  --output_dir outputs/MOT17-train \
  --filters standard nsa stf gstakf \
  --noise_levels high
```

Tracking outputs can be evaluated with the official TrackEval toolkit. Randomized perturbation results may differ slightly between runs, while the relative robustness trend should remain comparable.

## License

This project retains the GNU General Public License v3.0 of the original DeepSORT repository.
It is based on the original [DeepSORT implementation](https://github.com/nwojke/deep_sort).

## Citation

Please cite the original DeepSORT paper and the GSTAKF paper when using this repository:

```bibtex
@article{Yang2026GSTAKF,
  author  = {Hongtao Yang and Qiushi Yi and Xiulan Li and Hetian Li and
             Junyang Wang and Liming Bai},
  title   = {An improved GSTAKF-based pedestrian tracking method for abrupt state estimation},
  journal = {Journal of King Saud University Computer and Information Sciences},
  volume  = {38},
  number  = {888},
  year    = {2026},
  doi     = {10.1007/s44443-026-01309-x},
  url     = {https://doi.org/10.1007/s44443-026-01309-x}
}
```
