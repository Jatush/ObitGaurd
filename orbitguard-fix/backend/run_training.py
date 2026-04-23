#!/usr/bin/env python3
"""
OrbitGuard PPO Training Script (PyTorch)
=========================================
Trains the PPO agent on TLE-derived collision-avoidance scenarios.

Usage:
    python run_training.py [--episodes N] [--force] [--quiet]

Options:
    --episodes N    Training episodes (default: 500)
    --force         Force re-train even if saved model exists
    --quiet         Suppress progress output
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--force",  action="store_true")
    parser.add_argument("--quiet",  action="store_true")
    args = parser.parse_args()

    from core.ppo_agent import (
        PPOAgent, OrbitEnv, DEVICE, get_or_train_agent,
        evaluate_agent, RandomAgent, GreedyProgradeAgent, DebrisDirectedAgent,
    )

    verbose = not args.quiet
    if verbose:
        print("=" * 65)
        print("  OrbitGuard PPO Training  (PyTorch backend)")
        print(f"  Device : {DEVICE}")
        print("=" * 65)

    agent, metrics = get_or_train_agent(
        n_episodes=args.episodes,
        force_retrain=args.force,
        verbose=verbose,
    )

    if verbose:
        _print_report(metrics)

    out = os.path.join(os.path.dirname(__file__), "data", "training_metrics.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    if verbose:
        print(f"\n[✓] Full metrics → {out}")


def _print_report(m: dict):
    sep = "-" * 65

    print(f"\n{'TRAINING SUMMARY':^65}")
    print(sep)
    tr = m.get("training", {})
    print(f"  Episodes trained   : {tr.get('n_episodes')}")
    print(f"  Total env steps    : {tr.get('total_steps')}")
    print(f"  Training time      : {tr.get('training_time_s', 0):.1f}s")
    print(f"  Throughput         : {tr.get('episodes_per_sec', 0):.1f} ep/s")
    print(f"  Device             : {tr.get('device')}")

    print(f"\n{'REWARD METRICS':^65}")
    print(sep)
    rw = m.get("reward", {})
    print(f"  Mean reward        : {rw.get('mean', 0):+.4f}")
    print(f"  Std  reward        : {rw.get('std',  0):.4f}")
    print(f"  Min / Max          : {rw.get('min',  0):+.4f} / {rw.get('max', 0):+.4f}")
    print(f"  Final-50 mean      : {rw.get('final_50_mean', 0):+.4f}")
    print(f"  Improvement        : {rw.get('improvement', 0):+.4f}")

    print(f"\n{'LOSS METRICS':^65}")
    print(sep)
    ls = m.get("loss", {})
    print(f"  Policy loss (mean) : {ls.get('policy_loss_mean', 0):.6f}")
    print(f"  Policy loss (final): {ls.get('policy_loss_final', 0):.6f}")
    print(f"  Value  loss (mean) : {ls.get('value_loss_mean', 0):.6f}")
    print(f"  Value  loss (final): {ls.get('value_loss_final', 0):.6f}")
    print(f"  Entropy (mean)     : {ls.get('entropy_mean', 0):.6f}")
    print(f"  Approx KL (mean)   : {ls.get('approx_kl_mean', 0):.6f}")
    print(f"  Clip fraction      : {ls.get('clip_fraction', 0):.4f}")
    print(f"  Converged          : {'YES ✓' if ls.get('converged') else 'NO'}")

    print(f"\n{'TASK METRICS (RL)':^65}")
    print(sep)
    tm = m.get("task_metrics", {})
    print(f"  Success rate       : {tm.get('success_rate', 0)*100:.1f}%")
    print(f"  Collision rate     : {tm.get('collision_rate', 0)*100:.1f}%")
    print(f"  Success definition : {tm.get('success_definition', 'final_miss_km >= 5.0')}")
    print(f"  Collision definition: {tm.get('collision_definition', 'final_miss_km < 1.0')}")

    print(f"\n{'MISS DISTANCE METRICS':^65}")
    print(sep)
    md = m.get("miss_distance", {})
    print(f"  Mean miss distance : {md.get('mean_km',   0):.3f} km")
    print(f"  Std                : {md.get('std_km',    0):.3f} km")
    print(f"  Median             : {md.get('median_km', 0):.3f} km")
    print(f"  Max achieved       : {md.get('max_km',    0):.3f} km")
    pcts = md.get("percentiles", {})
    if pcts:
        print(f"  Percentiles (km)   : "
              f"p10={pcts.get('p10',0):.1f}  "
              f"p50={pcts.get('p50',0):.1f}  "
              f"p90={pcts.get('p90',0):.1f}  "
              f"p99={pcts.get('p99',0):.1f}")

    print(f"\n{'EVALUATION (deterministic policy)':^65}")
    print(sep)
    ev = m.get("evaluation", {})
    print(f"  Eval episodes      : {ev.get('n_eval_episodes', 0)}")
    print(f"  Mean reward        : {ev.get('mean_reward',   0):+.4f}")
    print(f"  Success rate       : {ev.get('success_rate',  0)*100:.1f}%")
    print(f"  Mean miss          : {ev.get('mean_miss_km',  0):.3f} km")

    print(f"\n{'MODEL COMPARISON':^65}")
    print(sep)
    mc = m.get("model_comparison", {})
    if mc:
        print(f"  {'Model':<22} {'Miss km':>9} {'Success%':>10}")
        print(f"  {'-'*61}")
        labels = {
            "random":          "Random",
            "greedy_prograde": "Greedy (prograde)",
            "debris_directed": "Debris-directed",
            "ppo_trained":     "PPO (trained) ★",
        }
        for name, ms in mc.items():
            print(f"  {labels.get(name, name):<22} "
                  f"{ms.get('mean_miss_km', 0):>9.3f} "
                  f"{ms.get('success_rate', 0)*100:>9.1f}%")

    print(f"\n{'PPO CONFIG (PyTorch)':^65}")
    print(sep)
    pp = m.get("ppo", {})
    print(f"  Framework          : {pp.get('framework','')}")
    print(f"  Algorithm          : {pp.get('algorithm','')}")
    print(f"  Device             : {pp.get('device','')}")
    print(f"  Policy arch        : {pp.get('policy_arch','')}")
    print(f"  Critic arch        : {pp.get('value_arch','')}")
    print(f"  Clip epsilon ε     : {pp.get('clip_epsilon','')}")
    print(f"  GAE λ              : {pp.get('gae_lambda','')}")
    print(f"  LR actor / critic  : {pp.get('lr_actor','')} / {pp.get('lr_critic','')}")
    print(f"  Optimiser          : {pp.get('optimizer','')}")
    print(f"  Weight init        : {pp.get('weight_init','')}")
    print(f"  Actor params       : {pp.get('n_parameters_actor', 0):,}")
    print(f"  Critic params      : {pp.get('n_parameters_critic', 0):,}")
    print(f"  Total params       : {pp.get('n_parameters_total', 0):,}")
    print(f"  Loss formula       : {pp.get('loss_formula','')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
