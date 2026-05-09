"""Entry point: collect data (optional) then fine-tune Qwen2-VL on GIS flood labels."""
import argparse
import sys
from pathlib import Path

# Ensure ml_serving/ is on sys.path so data_pipeline and training are importable
# regardless of which directory the script is invoked from.
sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2-VL-7B on GIS flood data")
    sub = parser.add_subparsers(dest="command", required=True)

    # -- live-collect sub-command (10 s intervals, 4 h duration) ----------
    lc = sub.add_parser(
        "live-collect",
        help="Collect live GIS samples at 10-second intervals for 4 hours (~1440 samples)",
    )
    lc.add_argument("--duration-hours", type=float, default=4.0, help="Total collection duration in hours")
    lc.add_argument("--interval", type=int, default=10, help="Seconds between samples (default 10)")
    lc.add_argument("--satellite-interval", type=int, default=30, help="Satellite refresh interval in minutes")
    lc.add_argument("--lat", type=float, default=6.9271, help="Target latitude (default: Colombo)")
    lc.add_argument("--lon", type=float, default=79.8612, help="Target longitude")
    lc.add_argument("--no-append", action="store_true", help="Start fresh dataset (discard existing index)")

    # -- collect sub-command (original, with full Sentinel download per sample) --
    col = sub.add_parser("collect", help="Collect training samples (slow, full Sentinel per sample)")
    col.add_argument("--n-samples", type=int, default=10)
    col.add_argument("--interval", type=int, default=300, help="Seconds between samples")
    col.add_argument("--lat", type=float, default=6.9271)
    col.add_argument("--lon", type=float, default=79.8612)

    # -- enrich sub-command -----------------------------------------------
    enr = sub.add_parser("enrich", help="Generate training labels from GIS data")
    enr.add_argument("--dataset-index", required=True)

    # -- train sub-command ------------------------------------------------
    trn = sub.add_parser("train", help="Fine-tune Qwen2-VL with LoRA")
    trn.add_argument("--dataset-index", required=True)
    trn.add_argument("--output-dir", default="checkpoints")
    trn.add_argument("--epochs", type=int, default=3)
    trn.add_argument("--batch-size", type=int, default=2)
    trn.add_argument("--lr", type=float, default=2e-4)
    trn.add_argument("--eval-split", type=float, default=0.1)
    trn.add_argument("--enrich-labels", action="store_true")
    trn.add_argument("--resume-from", default=None)

    args = parser.parse_args()

    if args.command == "live-collect":
        from data_pipeline.data_collector import fast_collect_live
        fast_collect_live(
            duration_hours=args.duration_hours,
            interval_seconds=args.interval,
            satellite_interval_minutes=args.satellite_interval,
            lat=args.lat,
            lon=args.lon,
            append_to_existing=not args.no_append,
        )

    elif args.command == "collect":
        from data_pipeline.data_collector import collect_dataset
        collect_dataset(n_samples=args.n_samples, interval_seconds=args.interval)

    elif args.command == "enrich":
        from training.label_generator import enrich_dataset_labels
        enrich_dataset_labels(args.dataset_index)

    elif args.command == "train":
        if args.enrich_labels:
            from training.label_generator import enrich_dataset_labels
            enrich_dataset_labels(args.dataset_index)
        from training.trainer import train
        train(
            dataset_index=args.dataset_index,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            eval_split=args.eval_split,
            resume_from=args.resume_from,
        )


if __name__ == "__main__":
    main()
