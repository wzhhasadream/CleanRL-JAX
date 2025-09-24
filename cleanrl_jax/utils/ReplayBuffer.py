import collections
from typing import Tuple, Union, Dict, Any, Optional

import numpy as np
import jax.numpy as jnp
import gymnasium as gym



Batch = Dict[str, jnp.ndarray]


def create_batch(
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    next_observations: np.ndarray
) -> Batch:
    """Create a batch dictionary for JAX"""
    return {
        'observations': jnp.array(observations),
        'actions': jnp.array(actions),
        'rewards': jnp.array(rewards.reshape(-1, 1)),
        'dones': jnp.array(dones.reshape(-1, 1)),
        'next_observations': jnp.array(next_observations)
    }




class ReplayBuffer:
    """
    Replay buffer for online RL with multi-env support and optional linear bias sampling.

    Args:
        obs_shape: Shape of observations (int or tuple)
        action_shape: Shape of actions (int or tuple)
        max_size: Maximum buffer size
        n_envs: Number of parallel environments
        linear_decay_steps: Controls sampling bias direction:
            - 0: uniform sampling (no bias)
            - >0: newer-biased (prefer recent experiences) 
            - <0: older-biased (prefer older experiences)
        min_weight: Minimum weight for biased experiences (0.1 = 10% of maximum weight)
    """

    def __init__(
        self,
        obs_shape: Union[int, tuple],
        action_shape: Union[int, tuple],
        max_size: int = int(1e6),
        n_envs: int = 1,
        linear_decay_steps: int = 0,
        min_weight: float = 0.1
    ):
        self.max_size = max_size
        self.n_envs = n_envs
        self.ptr = 0
        self.size = 0

        # Linear bias parameters
        self._raw_linear_decay_steps = linear_decay_steps  # Keep original sign
        self.linear_decay_steps = abs(linear_decay_steps)  # Use absolute value for calculations
        self.min_weight = min_weight
        self.current_time = 0

        # Validate parameters
        assert 0 <= min_weight <= 1, f"min_weight must be in [0, 1], got {min_weight}"

        # Handle both int and tuple for obs_shape
        if isinstance(obs_shape, int):
            self.obs_shape = (obs_shape,)
        else:
            self.obs_shape = obs_shape

        # Handle both int and tuple for action_shape
        if isinstance(action_shape, int):
            self.action_shape = (action_shape,)
        else:
            self.action_shape = action_shape

        # Initialize buffers with proper shapes
        self.observations = np.zeros((max_size, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((max_size, *self.action_shape), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.next_observations = np.zeros((max_size, *self.obs_shape), dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)
        self.timestamps = np.zeros(max_size, dtype=np.int64)  # Track when each sample was added

    @classmethod
    def from_env(
        cls,
        env: gym.vector.VectorEnv,
        max_size: int = int(1e6),
        linear_decay_steps: int = 0,
        min_weight: float = 0.1
    ) -> 'ReplayBuffer':
        """Create ReplayBuffer from environment - convenience method."""
        obs_shape = env.single_observation_space.shape
        action_shape = env.single_action_space.shape
        n_envs = getattr(env, 'num_envs', 1)
        return cls(obs_shape, action_shape, max_size, n_envs, linear_decay_steps, min_weight)
    
    def add(self, obs: np.ndarray, action: np.ndarray, reward: Union[float, np.ndarray],
            next_obs: np.ndarray, done: Union[bool, np.ndarray], infos: Optional[list] = None):
        """Add transition(s) to the buffer. Supports both single and multi-env."""

        batch_timestamp = self.current_time

        # Handle multi-env case
        if self.n_envs > 1:
            batch_size = len(obs) if hasattr(obs, '__len__') else 1
            for i in range(batch_size):
                self._add_single(
                    obs[i] if batch_size > 1 else obs,
                    action[i] if batch_size > 1 else action,
                    reward[i] if hasattr(reward, '__len__') else reward,
                    next_obs[i] if batch_size > 1 else next_obs,
                    done[i] if hasattr(done, '__len__') else done,
                    timestamp=batch_timestamp
                )
        else:
            self._add_single(obs, action, reward, next_obs, done, timestamp=batch_timestamp)

        # Always increment time to track sample age
        self.current_time += 1

    def _add_single(self, obs: np.ndarray, action: np.ndarray, reward: float,
                   next_obs: np.ndarray, done: bool, timestamp: Optional[int] = None):
        """Add a single transition to the buffer."""
        self.observations[self.ptr] = obs.reshape(self.obs_shape)
        self.actions[self.ptr] = action.reshape(self.action_shape)
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs.reshape(self.obs_shape)
        self.dones[self.ptr] = float(done)

        # Always record timestamp to track sample age
        self.timestamps[self.ptr] = timestamp if timestamp is not None else self.current_time

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size: int) -> Batch:
        """
        Sample a batch from the buffer with optional linear decay weighting.

        Args:
            batch_size: Number of samples to draw

        Returns:
            Batch dictionary containing sampled experiences
        """
        if self.size < batch_size:
            raise ValueError(f"Buffer size ({self.size}) < batch size ({batch_size})")

        if self._raw_linear_decay_steps == 0:
            # Uniform sampling (no bias)
            indx = np.random.randint(0, self.size, size=batch_size)
        else:
            # Linear bias sampling
            indx = self._sample_with_bias(batch_size)

        return create_batch(
            observations=self.observations[indx],
            actions=self.actions[indx],
            rewards=self.rewards[indx],
            dones=self.dones[indx],
            next_observations=self.next_observations[indx]
        )

    def _sample_with_bias(self, batch_size: int) -> np.ndarray:
        """
        Sample indices with linear bias weighting.

        Bias direction is controlled by _raw_linear_decay_steps sign:
          >0 : newer-biased (weights decrease linearly with age)
          <0 : older-biased (weights increase linearly with age)

        Args:
            batch_size: Number of indices to sample

        Returns:
            Array of sampled indices
        """
        valid_timestamps = self.timestamps[:self.size]
        age = self.current_time - valid_timestamps  # Age of each experience

        if self._raw_linear_decay_steps > 0:
            # Newer-biased: weight = max(min_weight, 1 - age/steps)
            weights = np.maximum(self.min_weight, 1.0 - age / self.linear_decay_steps)
        else:
            # Older-biased: weight = min(1.0, min_weight + age/steps)
            weights = np.minimum(1.0, self.min_weight + age / self.linear_decay_steps)


        probabilities = weights / weights.sum()

        # Sample indices according to probabilities
        return np.random.choice(self.size, size=batch_size, p=probabilities)

    def ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self.size >= batch_size

    def reset(self):
        """Reset the buffer."""
        self.ptr = 0
        self.size = 0
        self.current_time = 0
        if self.linear_decay_steps != 0:
            self.timestamps.fill(0)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        decay_info = f", decay_steps={self.linear_decay_steps}, min_weight={self.min_weight}" if self.linear_decay_steps > 0 else ""
        return f"ReplayBuffer(size={self.size}/{self.max_size}, obs_shape={self.obs_shape}, n_envs={self.n_envs}{decay_info})"
