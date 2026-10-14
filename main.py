"""Command-line entry point.

  python main.py all        # simulate -> clean -> train -> detect -> charts (everything)
  python main.py simulate   # generate datasets only
  python main.py train      # train + evaluate models
  python main.py detect     # run detection on unseen traffic and raise alerts
  python main.py visualize  # (re)draw charts
  python main.py serve      # start the REST API
"""
import argparse

from src.config import LIVE_DATA, RAW_DATA, DATA_DIR


def simulate(args):
    from src.data_simulator import generate_traffic
    DATA_DIR.mkdir(exist_ok=True)
    train = generate_traffic(n_samples=args.samples, seed=42)
    train.to_csv(RAW_DATA, index=False)
    live = generate_traffic(n_samples=3000, attack_ratio=0.20, seed=2026,
                            start="2026-02-01", messy=False)
    live.to_csv(LIVE_DATA, index=False)
    print(f"[simulate] training data: {RAW_DATA} {train.shape}")
    print(f"[simulate] live traffic : {LIVE_DATA} {live.shape}")
    print("\nPreview:\n", train.head().to_string())


def train(args):
    from src.preprocessing import clean_data, load_dataset
    from src.train import train_all
    train_all(clean_data(load_dataset(args.data or RAW_DATA)))


def detect(args):
    import pandas as pd
    from src.detect import run_detection
    df = pd.read_csv(args.live or LIVE_DATA, parse_dates=["timestamp"])
    run_detection(df, replay_delay=args.delay)


def visualize(args):
    from src.visualize import generate_all
    generate_all()


def serve(args):
    from src.app import app
    app.run(host="127.0.0.1", port=5000)


def run_all(args):
    args.data = None
    simulate(args); train(args); args.live = None; args.delay = 0.0
    detect(args); visualize(args)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="AI-Powered Cybersecurity Threat Detection")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in [("all", run_all), ("simulate", simulate), ("train", train),
                     ("detect", detect), ("visualize", visualize), ("serve", serve)]:
        sp = sub.add_parser(name); sp.set_defaults(func=fn)
        if name in ("all", "simulate"):
            sp.add_argument("--samples", type=int, default=20000, help="training rows to simulate")
        if name == "train":
            sp.add_argument("--data", help="path to your own CSV (same columns)")
        if name == "detect":
            sp.add_argument("--live", help="CSV of traffic to analyse")
            sp.add_argument("--delay", type=float, default=0.0, help="seconds between alerts (demo replay)")
    a = p.parse_args(); a.func(a)
