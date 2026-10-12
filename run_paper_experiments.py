import argparse
from pathlib import Path

from deep_sort_app import run


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mot_dir", required=True)
    parser.add_argument("--detection_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--filters",
        nargs="+",
        choices=["standard", "nsa", "stf", "gstakf"],
        default=["standard", "nsa", "stf", "gstakf"],
    )
    parser.add_argument(
        "--noise_levels",
        nargs="+",
        choices=["none", "medium", "high"],
        default=["none", "medium", "high"],
    )
    parser.add_argument("--min_confidence", type=float, default=0.3)
    parser.add_argument("--nms_max_overlap", type=float, default=1.0)
    parser.add_argument("--min_detection_height", type=int, default=0)
    parser.add_argument("--max_cosine_distance", type=float, default=0.2)
    parser.add_argument("--nn_budget", type=int, default=100)
    parser.add_argument("--noise_seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    mot_dir = Path(args.mot_dir)
    detection_dir = Path(args.detection_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_dirs = sorted(
        path for path in mot_dir.iterdir()
        if path.is_dir() and (path / "seqinfo.ini").exists()
    )
    for filter_type in args.filters:
        for noise_level in args.noise_levels:
            data_dir = (
                output_dir
                / (filter_type + "_" + noise_level)
                / "data"
            )
            data_dir.mkdir(parents=True, exist_ok=True)
            for sequence_dir in sequence_dirs:
                detection_file = (
                    detection_dir / (sequence_dir.name + ".npy")
                )
                if not detection_file.exists():
                    raise FileNotFoundError(detection_file)
                output_file = data_dir / (sequence_dir.name + ".txt")
                run(
                    str(sequence_dir),
                    str(detection_file),
                    str(output_file),
                    args.min_confidence,
                    args.nms_max_overlap,
                    args.min_detection_height,
                    args.max_cosine_distance,
                    args.nn_budget,
                    False,
                    filter_type=filter_type,
                    noise_level=noise_level,
                    noise_seed=args.noise_seed,
                )


if __name__ == "__main__":
    main()
