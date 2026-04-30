import argparse
from dataclasses import fields, replace

from config import Config


def _add_config_overrides(parser: argparse.ArgumentParser):
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--lr", type=float, dest="LEARNING_RATE")
    parser.add_argument("--batch-size", type=int, dest="BATCH_SIZE")
    parser.add_argument("--seed", type=int, dest="SEED")
    parser.add_argument("--run-name", type=str, dest="RUN_NAME")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], dest="DEVICE")
    parser.add_argument("--resume", type=str, dest="RESUME_FROM")


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    overrides = {}
    field_names = {f.name for f in fields(Config)}
    if args.__dict__.get("total_steps") is not None:
        overrides["TOTAL_STEPS"] = args.total_steps
    for k, v in vars(args).items():
        if k in field_names and v is not None:
            overrides[k] = v
    return replace(cfg, **overrides) if overrides else cfg


def main():
    parser = argparse.ArgumentParser(description="DQN Breakout — entry point")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="train the agent")
    _add_config_overrides(p_train)

    p_eval = sub.add_parser("eval", help="evaluate a checkpoint")
    p_eval.add_argument("--checkpoint", type=str, default="checkpoints/dqn_latest.pt")
    p_eval.add_argument("--episodes", type=int, default=10)
    p_eval.add_argument("--epsilon", type=float, default=0.05)
    _add_config_overrides(p_eval)

    p_watch = sub.add_parser("watch", help="render the agent playing")
    p_watch.add_argument("--checkpoint", type=str, default="checkpoints/dqn_latest.pt")
    p_watch.add_argument("--episodes", type=int, default=3)
    p_watch.add_argument("--epsilon", type=float, default=0.05)
    _add_config_overrides(p_watch)

    p_baseline = sub.add_parser("baseline", help="train SB3 DQN baseline")
    _add_config_overrides(p_baseline)

    args = parser.parse_args()
    cfg = _apply_overrides(Config(), args)

    if args.cmd == "train":
        from train import train
        train(cfg)
    elif args.cmd == "eval":
        from evaluate import evaluate
        evaluate(cfg, args.checkpoint, episodes=args.episodes, epsilon=args.epsilon)
    elif args.cmd == "watch":
        from evaluate import evaluate
        evaluate(cfg, args.checkpoint, episodes=args.episodes, epsilon=args.epsilon,
                 render_mode="human")
    elif args.cmd == "baseline":
        from baseline_comparison import run_baseline
        run_baseline(cfg)


if __name__ == "__main__":
    main()
