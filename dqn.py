"""Double DQN, replay, and environment setup shared by both experiments."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AutoresetMode
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    game: str
    env_id: str
    gamma: float
    episodes: int
    num_envs: int
    capacity: int
    learning_starts: int
    train_every: int
    batch_size: int
    learning_rate: float
    target_every: int
    epsilon_steps: int
    epsilon_end: float
    n_step: int
    reward_scale: float
    life_loss_terminal: bool
    validation_every: int
    validation_episodes: int
    seed: int = 42
    failure_reward: float | None = None
    normalize_cartpole: bool = False

    @property
    def pixels(self):
        return self.game == "mspacman"


def config_for(game):
    if game == "cartpole":
        return Config(
            game=game, env_id="CartPole-v1", gamma=0.95, episodes=500,
            num_envs=8, capacity=50_000, learning_starts=1_024,
            train_every=4, batch_size=64, learning_rate=0.0005,
            target_every=250, epsilon_steps=12_000, epsilon_end=0.03,
            n_step=3, reward_scale=1, life_loss_terminal=False, seed=137,
            validation_every=20, validation_episodes=80, normalize_cartpole=True,
        )
    if game == "mspacman":
        return Config(
            game=game, env_id="ALE/MsPacman-v5", gamma=0.99, episodes=4_000,
            num_envs=24, capacity=75_000, learning_starts=12_000,
            train_every=12, batch_size=64, learning_rate=0.00015,
            target_every=800, epsilon_steps=550_000, epsilon_end=0.05,
            n_step=3, reward_scale=10, life_loss_terminal=True,
            validation_every=250, validation_episodes=24,
        )
    raise ValueError(f"Unknown experiment {game!r}")


def make_env(config, num_envs, workers, *, sticky=None, frame_limit=None):
    if config.pixels:
        import ale_py

        gym.register_envs(ale_py)
        options = {}
        if sticky is not None:
            options["repeat_action_probability"] = sticky
        if frame_limit is not None:
            options["max_num_frames_per_episode"] = frame_limit
        return gym.make_vec(
            config.env_id, num_envs=num_envs, vectorization_mode="vector_entry_point",
            num_threads=min(workers, num_envs), autoreset_mode=AutoresetMode.NEXT_STEP,
            reward_clipping=False, life_loss_info=False, episodic_life=False,
            use_fire_reset=False, **options,
        )
    return gym.make_vec(
        config.env_id, num_envs=num_envs, vectorization_mode="sync",
        vector_kwargs={"autoreset_mode": AutoresetMode.NEXT_STEP},
    )


class QNetwork(nn.Module):
    def __init__(self, observation_shape, num_actions, *, normalize_cartpole=False):
        super().__init__()
        self.pixels = len(observation_shape) == 3
        if self.pixels:
            self.encoder = nn.Sequential(
                nn.Conv2d(observation_shape[0], 16, kernel_size=8, stride=4), nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3), nn.ReLU(), nn.Flatten(),
            )
            with torch.no_grad():
                features = self.encoder(torch.zeros(1, *observation_shape)).shape[1]
        else:
            self.encoder = nn.Sequential(nn.Linear(observation_shape[0], 128), nn.ReLU())
            features = 128
        self.normalize_cartpole = normalize_cartpole
        if normalize_cartpole:
            self.register_buffer("cartpole_scale", torch.tensor([1 / 2.4, 0.5, 15 / np.pi, 0.5]),
                                 persistent=False)
        width = 128
        self.value = nn.Sequential(nn.Linear(features, width), nn.ReLU(), nn.Linear(width, 1))
        self.advantage = nn.Sequential(
            nn.Linear(features, width), nn.ReLU(), nn.Linear(width, num_actions)
        )

    def forward(self, states):
        if self.pixels:
            states = states.float() / 255.0
        elif self.normalize_cartpole:
            states = states * self.cartpole_scale
        features = self.encoder(states)
        advantages = self.advantage(features)
        return self.value(features) + (advantages - advantages.mean(dim=1, keepdim=True))


class Replay:
    """Uniform sampling with replacement from a fixed-size transition ring."""

    def __init__(self, capacity, shape, dtype):
        self.states = np.empty((capacity, *shape), dtype=dtype)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.returns = np.empty(capacity, dtype=np.float32)
        self.discounts = np.empty(capacity, dtype=np.float32)
        self.capacity = capacity
        self.cursor = 0
        self.size = 0

    def append(self, state, action, rewards, next_state, discount):
        index = self.cursor
        self.states[index] = state
        self.next_states[index] = next_state
        self.actions[index] = action
        self.returns[index] = rewards
        self.discounts[index] = discount
        self.cursor = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, size, rng, device):
        indices = rng.integers(self.size, size=size)
        arrays = (self.states, self.actions, self.returns, self.next_states, self.discounts)
        return tuple(torch.from_numpy(array[indices]).to(device) for array in arrays)


class NStep:
    """Build returns within one trajectory and drain its tail at a boundary."""

    def __init__(self, steps, gamma):
        self.steps = steps
        self.gamma = gamma
        self.pending = deque()

    def append(self, replay, state, action, reward, next_state, *, boundary, terminal):
        self.pending.append((state, action, reward))
        while len(self.pending) >= self.steps or (boundary and self.pending):
            discounted_sum = sum(r * self.gamma**i for i, (_, _, r) in enumerate(self.pending))
            discount = 0.0 if terminal else self.gamma ** len(self.pending)
            oldest_state, oldest_action, _ = self.pending.popleft()
            replay.append(oldest_state, oldest_action, discounted_sum, next_state, discount)


def double_targets(rewards, discounts, online_next, target_next):
    actions = online_next.argmax(dim=1, keepdim=True)
    return rewards + discounts * target_next.gather(1, actions).squeeze(1)


def learning_reward(config, raw_reward, terminated):
    if terminated and config.failure_reward is not None:
        raw_reward = config.failure_reward
    return raw_reward / config.reward_scale


def optimize(online, target, optimizer, batch):
    states, actions, rewards, next_states, discounts = batch
    chosen_q = online(states).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        y = double_targets(rewards, discounts, online(next_states), target(next_states))
    loss = F.smooth_l1_loss(chosen_q, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    optimizer.step()
    return loss.detach()


def pick_device(requested, threads):
    torch.set_num_threads(threads)
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
    return device


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hardware_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple Metal GPU"
    return "CPU"


@torch.inference_mode()
def policy_values(model, obs, device, freeze_motion=False):
    if freeze_motion:
        obs = np.repeat(obs[:, -1:], obs.shape[1], axis=1)
    return model(torch.from_numpy(obs).to(device)).cpu().numpy()


def save_checkpoint(path, model, config, shape, actions, episode, steps):
    from dataclasses import asdict

    temp = path.with_suffix(".tmp")
    torch.save({
        "weights": model.state_dict(), "config": asdict(config),
        "shape": shape, "actions": actions, "episode": episode, "steps": steps,
    }, temp)
    temp.replace(path)


def load_checkpoint(path: Path, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = Config(**payload["config"])
    model = QNetwork(payload["shape"], payload["actions"], normalize_cartpole=config.normalize_cartpole).to(device)
    model.load_state_dict(payload["weights"])
    return model.eval(), config, payload
