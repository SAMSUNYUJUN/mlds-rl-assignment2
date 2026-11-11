"""Train and evaluate the assignment's two DQN agents."""

import argparse
import csv
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import gymnasium
import numpy as np
import torch

from dqn import (
    NStep, QNetwork, Replay, config_for, hardware_name, learning_reward, load_checkpoint, make_env,
    optimize, pick_device, policy_values, save_checkpoint, seed_torch,
)


SELECTION_SEED = 100_000
EVALUATION_SEED = 200_000
EVALUATION_EPISODES = 500
TRAIN_COLUMNS = (
    "episode", "steps", "updates", "env_index", "reward", "length", "max_q", "epsilon",
    "terminated", "truncated", "life_losses", "elapsed_seconds",
)
EVAL_COLUMNS = (
    "env_seed", "episode_in_env", "reward", "length", "terminated", "truncated", "life_losses",
)


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def write_csv(path, columns, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summarize(rows):
    rewards = np.array([row["reward"] for row in rows], dtype=np.float64)
    return {"episodes": len(rows), "mean": float(rewards.mean()), "sd": float(rewards.std(ddof=0)),
            "median": float(np.median(rewards)), "min": float(rewards.min()),
            "max": float(rewards.max()), "terminated": sum(row["terminated"] for row in rows),
            "truncated": sum(row["truncated"] for row in rows),
            "balanced_to_500": sum(row["length"] == 500 and not row["terminated"] for row in rows)}


def rollouts(env, model, device, *, count, seed, freeze_motion=False):
    """Collect a fixed episode quota per lane using independent environment seeds."""
    obs, info = env.reset(seed=seed)
    num_envs = env.num_envs
    indices = np.arange(num_envs)
    quotas = count // num_envs + (indices < count % num_envs)
    n_actions = env.single_action_space.n
    rng = np.random.default_rng(seed + 1_000_000)
    rngs = [np.random.default_rng(s) for s in rng.integers(2**32, size=num_envs)] if model is None else []
    previous_done = np.zeros(num_envs, dtype=bool)
    episode_numbers = np.zeros(num_envs, dtype=int)
    returns = np.zeros(num_envs)
    lengths = np.zeros(num_envs, dtype=int)
    losses = np.zeros(num_envs, dtype=int)
    lives = info.get("lives", np.zeros(num_envs, dtype=int))
    rows = []
    while len(rows) < count:
        active = episode_numbers < quotas
        valid = active & ~previous_done
        actions = np.zeros(num_envs, dtype=np.int64)
        if np.any(valid):
            if model is None:
                actions[valid] = [rngs[i].integers(n_actions) for i in np.flatnonzero(valid)]
            else:
                # Keep inference batch shapes stable on GPUs and ignore inactive lanes.
                actions = policy_values(model, obs, device, freeze_motion).argmax(axis=1)
        obs, rewards, terminated, truncated, info = env.step(actions)
        done = terminated | truncated
        next_lives = info.get("lives", lives)
        returns += np.where(valid, rewards, 0)
        lengths += valid
        losses += valid & (next_lives < lives)
        for i in np.flatnonzero(valid & done):
            episode_numbers[i] += 1
            rows.append(dict(zip(EVAL_COLUMNS, (
                int(seed + i), int(episode_numbers[i]), float(returns[i]), int(lengths[i]),
                int(terminated[i]), int(truncated[i]), int(losses[i]),
            ))))
            returns[i] = lengths[i] = losses[i] = 0
        previous_done = done
        lives = next_lives
    return sorted(rows, key=lambda row: (row["env_seed"], row["episode_in_env"]))


def train(config, out, device, workers):
    out.mkdir(parents=True, exist_ok=False)
    seed_torch(config.seed)
    rng = np.random.default_rng(config.seed)
    replay_rng = np.random.default_rng(config.seed + 1)
    env = make_env(config, config.num_envs, workers)
    validation_env = make_env(config, config.validation_episodes, workers)
    start_time = time.monotonic()
    try:
        obs, info = env.reset(seed=config.seed)
        shape = tuple(map(int, env.single_observation_space.shape))
        num_actions = int(env.single_action_space.n)
        online = QNetwork(shape, num_actions, normalize_cartpole=config.normalize_cartpole).to(device)
        target = QNetwork(shape, num_actions, normalize_cartpole=config.normalize_cartpole).to(device)
        target.eval().requires_grad_(False)
        target.load_state_dict(online.state_dict())
        optimizer = torch.optim.Adam(online.parameters(), lr=config.learning_rate)
        replay = Replay(config.capacity, shape, obs.dtype)
        queues = [NStep(config.n_step, config.gamma) for _ in range(config.num_envs)]
        save_checkpoint(out / "initial.pt", online, config, shape, num_actions, 0, 0)
        metadata = {
            "config": asdict(config), "device": str(device), "hardware": hardware_name(device),
            "platform": platform.platform(), "torch": torch.__version__, "gymnasium": gymnasium.__version__,
            "parameter_count": sum(p.numel() for p in online.parameters()), "shape": shape,
            "actions": num_actions, "selection_seed": SELECTION_SEED,
        }
        write_json(out / "metadata.json", metadata)
        print(f"Training {config.env_id} on {device} with {config.num_envs} lanes", flush=True)

        steps = updates = episode = 0
        next_update = config.learning_starts
        next_validation = config.validation_every
        best_validation = -np.inf
        scores = []
        previous_done = np.zeros(config.num_envs, dtype=bool)
        returns = np.zeros(config.num_envs)
        lengths = np.zeros(config.num_envs, dtype=int)
        losses = np.zeros(config.num_envs, dtype=int)
        peak_q = np.full(config.num_envs, -np.inf)
        lives = info.get("lives", np.zeros(config.num_envs, dtype=int))

        with (out / "training.csv").open("w", newline="") as train_file, \
                (out / "selection.csv").open("w", newline="") as selection_file:
            writer = csv.writer(train_file)
            writer.writerow(TRAIN_COLUMNS)
            selection_writer = csv.writer(selection_file)
            selection_writer.writerow(("episode", "steps", "greedy_mean", "greedy_sd", "new_best"))
            while episode < config.episodes:
                fraction = min(max(steps - config.learning_starts, 0) / config.epsilon_steps, 1.0)
                epsilon = 1.0 - fraction * (1.0 - config.epsilon_end)
                values = policy_values(online, obs, device)
                actions = values.argmax(axis=1)
                explore = rng.random(config.num_envs) < epsilon
                actions[explore] = rng.integers(num_actions, size=explore.sum())
                next_obs, rewards, terminated, truncated, info = env.step(actions)
                done = terminated | truncated
                next_lives = info.get("lives", lives)
                lost_life = next_lives < lives
                valid = ~previous_done
                for i in np.flatnonzero(valid):
                    terminal = terminated[i] or (config.life_loss_terminal and lost_life[i])
                    queues[i].append(
                        replay, obs[i], int(actions[i]), learning_reward(config, rewards[i], terminated[i]), next_obs[i],
                        boundary=done[i] or terminal, terminal=terminal,
                    )
                returns += np.where(valid, rewards, 0)
                lengths += valid
                losses += valid & lost_life
                np.maximum(peak_q, values.max(axis=1), out=peak_q, where=valid)
                steps += int(valid.sum())
                obs, previous_done, lives = next_obs, done, next_lives

                while steps >= next_update and replay.size >= config.batch_size:
                    optimize(online, target, optimizer, replay.sample(config.batch_size, replay_rng, device))
                    updates += 1
                    next_update += config.train_every
                    if updates % config.target_every == 0:
                        target.load_state_dict(online.state_dict())

                for i in np.flatnonzero(valid & done):
                    if episode == config.episodes:
                        break
                    episode += 1
                    scores.append(returns[i])
                    writer.writerow((episode, steps, updates, i, returns[i], lengths[i],
                                     f"{peak_q[i]:.6f}", f"{epsilon:.6f}", int(terminated[i]),
                                     int(truncated[i]), losses[i], f"{time.monotonic() - start_time:.2f}"))
                    returns[i] = lengths[i] = losses[i] = 0
                    peak_q[i] = -np.inf
                    if episode % 20 == 0 or episode == config.episodes:
                        train_file.flush()
                    if episode % 50 == 0 or episode == config.episodes:
                        elapsed = time.monotonic() - start_time
                        print(f"{episode:>5}/{config.episodes}  steps={steps:>8,}  eps={epsilon:.3f}  "
                              f"mean100={np.mean(scores[-100:]):7.1f}  {steps / elapsed:,.0f} steps/s",
                              flush=True)

                if episode >= next_validation or episode == config.episodes:
                    train_file.flush()
                    summary = summarize(rollouts(validation_env, online, device,
                                                 count=config.validation_episodes, seed=SELECTION_SEED))
                    is_best = summary["mean"] > best_validation
                    if is_best:
                        best_validation = summary["mean"]
                        save_checkpoint(out / "best.pt", online, config, shape, num_actions, episode, steps)
                    if not (out / "early.pt").exists():
                        save_checkpoint(out / "early.pt", online, config, shape, num_actions, episode, steps)
                    save_checkpoint(out / "last.pt", online, config, shape, num_actions, episode, steps)
                    selection_writer.writerow((episode, steps, summary["mean"], summary["sd"], int(is_best)))
                    selection_file.flush()
                    print(f"  validation at {episode}: {summary['mean']:.2f}"
                          f"{' (best)' if is_best else ''}", flush=True)
                    next_validation = (episode // config.validation_every + 1) * config.validation_every

        metadata.update(episodes=episode, steps=steps, updates=updates,
                        elapsed_seconds=time.monotonic() - start_time)
        write_json(out / "metadata.json", metadata)
    finally:
        env.close()
        validation_env.close()


def evaluate(run_dir, device, workers, *, count, lanes, seed):
    model, config, payload = load_checkpoint(run_dir / "best.pt", device)
    env = make_env(config, min(count, lanes), workers)
    try:
        rows = rollouts(env, model, device, count=count, seed=seed)
    finally:
        env.close()
    summary = summarize(rows)
    summary.update(checkpoint_episode=payload["episode"], checkpoint_steps=payload["steps"],
                   env_seed_start=seed, lanes=min(count, lanes), device=str(device))
    write_csv(run_dir / "evaluation.csv", EVAL_COLUMNS, rows)
    write_json(run_dir / "evaluation.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


def device_arguments(parser):
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--threads", type=positive_int, default=2, help="PyTorch CPU threads")
    parser.add_argument("--workers", type=positive_int, default=6, help="ALE emulator threads")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    trainer = subparsers.add_parser("train")
    trainer.add_argument("game", choices=("cartpole", "mspacman"))
    trainer.add_argument("--out", type=Path, required=True)
    trainer.add_argument("--seed", type=int)
    trainer.add_argument("--failure-reward", type=float, help="replace CartPole's physical failure reward")
    trainer.add_argument("--normalize-cartpole", action=argparse.BooleanOptionalAction, default=None)
    for name in ("episodes", "num-envs", "train-every", "epsilon-steps", "validation-every",
                 "validation-episodes", "capacity", "n-step"):
        trainer.add_argument(f"--{name}", type=positive_int)
    device_arguments(trainer)
    evaluator = subparsers.add_parser("evaluate")
    evaluator.add_argument("--run", type=Path, required=True)
    evaluator.add_argument("--episodes", type=positive_int, default=EVALUATION_EPISODES)
    evaluator.add_argument("--lanes", type=positive_int, default=25)
    evaluator.add_argument("--seed", type=int, default=EVALUATION_SEED)
    device_arguments(evaluator)
    args = parser.parse_args()
    device = pick_device(args.device, args.threads)
    if args.command == "train":
        config = config_for(args.game)
        if args.seed is not None:
            config.seed = args.seed
        if args.normalize_cartpole is not None:
            config.normalize_cartpole = args.normalize_cartpole
        if config.normalize_cartpole and config.pixels:
            parser.error("CartPole observation scaling can only be used with CartPole")
        if args.failure_reward is not None:
            if config.pixels or not np.isfinite(args.failure_reward) or args.failure_reward > 1:
                parser.error("failure reward must be finite, at most +1, and used with CartPole")
            config.failure_reward = args.failure_reward
        for attr in ("episodes", "num_envs", "train_every", "epsilon_steps", "validation_every",
                     "validation_episodes", "capacity", "n_step"):
            value = getattr(args, attr)
            if value is not None:
                setattr(config, attr, value)
        if config.capacity < config.batch_size:
            parser.error("replay capacity must be at least the minibatch size")
        train(config, args.out, device, args.workers)
    else:
        evaluate(args.run, device, args.workers, count=args.episodes, lanes=args.lanes, seed=args.seed)


if __name__ == "__main__":
    main()
