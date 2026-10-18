"""Train an experiment for every seed in ``cfg.seeds`` (or the ones given).

INT8 variants (``compress.int8: true``) are produced by post-training
quantization of their parent instead of training.

    python scripts/train.py --config configs/experiment/effnet_b0_ft.yaml
    python scripts/train.py --config configs/experiment/effnet_b0_ft.yaml --seed 1 --seed 2
"""

import argparse

from cropxai.compress.variants import is_int8, quantize_variant
from cropxai.train.loop import fit
from cropxai.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, action="append", dest="seeds", help="repeatable; default: cfg.seeds")
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = parser.parse_args()

    seeds = args.seeds or list(load_config(args.config, args.overrides).seeds)
    for seed in seeds:
        cfg = load_config(args.config, [*args.overrides, f"seed={seed}"])
        print(quantize_variant(cfg) if is_int8(cfg) else fit(cfg))


if __name__ == "__main__":
    main()
