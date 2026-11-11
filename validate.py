"""Run task-specific policy interventions and Bellman boundary experiments."""

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from dqn import NStep, Replay, double_targets, learning_reward, load_checkpoint, make_env, pick_device, policy_values
from run import device_arguments, positive_int, rollouts, write_csv, write_json


PROBE_SEED = 300_000


def compare_policies(run_dir, model, config, device, workers, count):
    initial, _, _ = load_checkpoint(run_dir / "initial.pt", device)
    early, _, early_info = load_checkpoint(run_dir / "early.pt", device)
    last, _, last_info = load_checkpoint(run_dir / "last.pt", device)
    policies = {"random": None, "initial": initial, "early": early, "last": last, "selected": model}
    if config.pixels:
        policies["frozen_frames"] = model
    env = make_env(config, count, workers)
    try:
        comparison = {}
        for name, policy in policies.items():
            results = rollouts(env, policy, device, count=count, seed=PROBE_SEED,
                               freeze_motion=name == "frozen_frames")
            comparison[name] = np.array([row["reward"] for row in results])
            print(f"  {name}: {comparison[name].mean():.2f} (n={count})", flush=True)
    finally:
        env.close()
    write_csv(run_dir / "validation_policies.csv", ("seed", *policies), [
        {"seed": PROBE_SEED + i, **{name: values[i] for name, values in comparison.items()}}
        for i in range(count)
    ])
    rng = np.random.default_rng(PROBE_SEED)
    indices = rng.integers(count, size=(10_000, count))
    comparisons = {}
    for name in comparison:
        if name != "selected":
            delta = comparison["selected"] - comparison[name]
            lo, hi = np.percentile(delta[indices].mean(axis=1), [2.5, 97.5])
            comparisons[name] = {"delta": float(delta.mean()), "ci_low": float(lo), "ci_high": float(hi),
                                 "wins": int((delta > 0).sum()), "ties": int((delta == 0).sum())}
    return {"episodes_per_policy": count, "first_seed": PROBE_SEED,
            "means": {name: float(values.mean()) for name, values in comparison.items()},
            "selected_minus": comparisons, "early_episode": early_info["episode"],
            "last_episode": last_info["episode"]}


def cartpole_grid(run_dir, model, config, device, workers):
    # At each fixed state, force left or right once and follow the same greedy policy thereafter.
    velocities, angles = np.meshgrid(np.linspace(-1.25, 1.25, 25), np.linspace(-0.205, 0.205, 25))
    starts = np.column_stack((np.zeros(angles.size), np.zeros(angles.size), angles.ravel(), velocities.ravel()))
    starts = starts.astype(np.float32)
    q_values = policy_values(model, starts, device)
    obs = np.repeat(starts, 2, axis=0)
    env = make_env(config, len(obs), workers)
    try:
        env.reset(seed=PROBE_SEED)
        for lane, state in zip(env.envs, obs):
            lane.unwrapped.state = tuple(map(float, state))
        actions = np.tile([0, 1], len(starts))
        alive = np.ones(len(obs), dtype=bool)
        lengths = np.zeros(len(obs), dtype=int)
        discount = 1.0
        returns = np.zeros(len(obs))
        for _ in range(500):
            obs, reward, terminated, truncated, _ = env.step(actions)
            returns += np.where(alive, discount * reward, 0)
            lengths += alive
            alive &= ~(terminated | truncated)
            if not np.any(alive):
                break
            actions = np.zeros(len(obs), dtype=np.int64)
            actions[alive] = policy_values(model, obs[alive], device).argmax(axis=1)
            discount *= config.gamma
    finally:
        env.close()

    returns = returns.reshape(-1, 2)
    lengths = lengths.reshape(-1, 2)
    q_gap = q_values[:, 1] - q_values[:, 0]
    realized_gap = returns[:, 1] - returns[:, 0]
    informative = np.abs(realized_gap) >= 0.10
    greedy = q_values.argmax(axis=1)
    regret = returns.max(axis=1) - returns[np.arange(len(starts)), greedy]
    names = ("angle_rad", "angular_velocity_rad_s", "q_right_minus_left", "return_right_minus_left",
             "steps_left", "steps_right")
    write_csv(run_dir / "validation_grid.csv", names, [
        dict(zip(names, row)) for row in zip(angles.ravel(), velocities.ravel(), q_gap, realized_gap,
                                             lengths[:, 0], lengths[:, 1])
    ])
    return {"grid_states": len(starts), "informative_states": int(informative.sum()),
            "informative_gap_minimum": 0.10, "action_agreement": int((regret[informative] == 0).sum()),
            "mean_regret": float(regret.mean()), "worst_regret": float(regret.max())}


def cartpole_boundaries(model, config, device):
    base = gym.make(config.env_id)
    capped = gym.make(config.env_id, max_episode_steps=3)
    try:
        state, _ = base.reset(seed=PROBE_SEED)
        capped.reset(seed=PROBE_SEED)
        for _ in range(3):
            previous_state = state
            action = int(policy_values(model, state[None], device)[0].argmax())
            state, raw_reward, terminated, truncated, _ = base.step(action)
            cut_state, cut_reward, cut_term, cut_trunc, _ = capped.step(action)
            np.testing.assert_array_equal(cut_state, state)
            assert raw_reward == cut_reward
        assert cut_trunc and not (cut_term or terminated or truncated)

        replay = Replay(config.n_step, state.shape, state.dtype)
        NStep(config.n_step, config.gamma).append(
            replay, previous_state, action, raw_reward, cut_state, boundary=True, terminal=False,
        )
        values = torch.from_numpy(policy_values(model, cut_state[None], device))
        cap_target = double_targets(torch.from_numpy(replay.returns[:1]),
                                    torch.from_numpy(replay.discounts[:1]), values, values).item()
        np.testing.assert_allclose(replay.discounts[0], config.gamma, rtol=1e-6)

        base.reset(seed=PROBE_SEED)
        near_failure = np.array([0, 0, 0.208, 0.4], dtype=np.float32)
        base.unwrapped.state = tuple(map(float, near_failure))
        final, failure_raw, terminated, truncated, _ = base.step(0)
        assert terminated and not truncated
        failure_reward = learning_reward(config, failure_raw, terminated)
        NStep(config.n_step, config.gamma).append(
            replay, near_failure, 0, failure_reward, final, boundary=True, terminal=True,
        )
        index = 1 % replay.capacity
        assert replay.discounts[index] == 0
        return {"cap_step": 3, "max_q_next_at_cap": float(values.max().item()),
                "target_at_cap": cap_target, "target_if_mislabeled_terminal": float(raw_reward),
                "failure_target": float(replay.returns[index])}
    finally:
        base.close()
        capped.close()


def atari_trace(model, config, device, workers, *, frame_limit=None):
    """Verify the replay tail on an emulator trajectory, using independent backward returns."""
    env = make_env(config, 1, workers, frame_limit=frame_limit)
    rows = []
    boundaries = []
    queue = NStep(config.n_step, config.gamma)
    replay = Replay(config.n_step, env.single_observation_space.shape, np.uint8)
    scores = []
    total = 0.0
    try:
        obs, info = env.reset(seed=PROBE_SEED + 100)
        lives = int(info["lives"][0])
        while True:
            values = policy_values(model, obs, device)[0]
            action = int(values.argmax())
            next_obs, rewards, term, trunc, info = env.step(np.array([action], dtype=np.int64))
            reward = float(rewards[0])
            total += reward
            next_lives = int(info["lives"][0])
            lost = next_lives < lives
            terminated, truncated = bool(term[0]), bool(trunc[0])
            terminal = terminated or (config.life_loss_terminal and lost)
            boundary = terminal or truncated
            rows.append({"step": len(rows) + 1, "action": action, "max_q": float(values.max()),
                         "reward": reward, "score": total, "lives": next_lives, "life_lost": int(lost),
                         "terminated": int(terminated), "truncated": int(truncated)})
            learning = learning_reward(config, reward, terminated)
            scores.append(learning)
            start = replay.cursor
            queue.append(replay, obs[0], action, learning, next_obs[0], boundary=boundary, terminal=terminal)
            if boundary:
                tail = scores[-config.n_step:]
                expected = []
                for value in reversed(tail):
                    expected.insert(0, value + config.gamma * (expected[0] if expected else 0))
                indices = (start + np.arange(len(tail))) % replay.capacity
                discounts = [0.0 if terminal else config.gamma**k for k in range(len(tail), 0, -1)]
                np.testing.assert_allclose(replay.returns[indices], expected, rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(replay.discounts[indices], discounts, rtol=1e-6)
                np.testing.assert_array_equal(replay.next_states[indices], np.repeat(next_obs, len(tail), axis=0))
                assert len(queue.pending) == 0
                boundaries.append({"step": len(rows), "kind": "game over" if terminated else
                                   "time limit" if truncated else "life loss", "raw_rewards": [
                                       row["reward"] for row in rows[-len(tail):]], "returns": expected,
                                   "bootstrap_multipliers": discounts, "lives": next_lives})
                scores.clear()
            obs, lives = next_obs, next_lives
            if terminated or truncated:
                return rows, boundaries
    finally:
        env.close()


def atari_boundaries(run_dir, model, config, device, workers):
    trace, boundaries = atari_trace(model, config, device, workers)
    first_boundary = boundaries[0]["step"]
    # End the comparison at a scored transition so its replay tail tests discounting too.
    cap_step = next((row["step"] for row in trace[:first_boundary - 1]
                     if row["reward"] and row["step"] >= config.n_step), min(40, first_boundary - 1))
    frame_limit = cap_step * gym.spec(config.env_id).kwargs["frameskip"]
    short, short_boundaries = atari_trace(model, config, device, workers, frame_limit=frame_limit)
    assert len(short) == cap_step and short[-1]["truncated"] and not short[-1]["terminated"]
    assert all(short_row["action"] == full_row["action"] and short_row["reward"] == full_row["reward"]
               for short_row, full_row in zip(short, trace))
    write_csv(run_dir / "validation_trace.csv", trace[0].keys(), trace)
    return {"boundaries": boundaries, "forced_cap": {**short_boundaries[-1], "frame_limit": frame_limit},
            "trace_seed": PROBE_SEED + 100}


def validate(run_dir, device, workers, count):
    model, config, payload = load_checkpoint(run_dir / "best.pt", device)
    print(f"Validating {config.env_id}, selected checkpoint {payload['episode']}", flush=True)
    results = compare_policies(run_dir, model, config, device, workers, count)
    results["selected_episode"] = payload["episode"]
    if config.pixels:
        results.update(atari_boundaries(run_dir, model, config, device, workers))
    else:
        results["grid"] = cartpole_grid(run_dir, model, config, device, workers)
        results["boundaries"] = cartpole_boundaries(model, config, device)
    write_json(run_dir / "validation.json", results)
    print(f"Saved validation evidence to {run_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--episodes", type=positive_int, default=48, help="paired seeds per policy")
    device_arguments(parser)
    args = parser.parse_args()
    validate(args.run, pick_device(args.device, args.threads), args.workers, args.episodes)


if __name__ == "__main__":
    main()
