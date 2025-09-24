import numpy as np
import gymnasium as gym
from typing import Tuple, Union, Any, Optional
from .ReplayBuffer import create_batch, Batch


class SumTree:
    """
    Sum Tree data structure for efficient prioritized sampling.
    
    This binary tree structure allows O(log n) sampling and priority updates.
    Each leaf node stores a priority value, and internal nodes store the sum
    of their children's priorities.
    
    Reference: Schaul et al. "Prioritized Experience Replay" (2015)
    """
    
    def __init__(self, capacity: int):
        """
        Initialize Sum Tree with given capacity.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        # Tree array: [internal_nodes, leaf_nodes]
        # For capacity n, we need n-1 internal nodes + n leaf nodes = 2n-1 total
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        # Data array to store actual experiences
        self.data = np.empty(capacity, dtype=object)
        # Current write position (circular)
        self.write_idx = 0
        # Number of stored experiences
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        """
        Propagate priority change up the tree to maintain sum property.
        
        Args:
            idx: Tree index where change occurred
            change: Amount of change in priority
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change
        
        if parent != 0:  # Not root
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """
        Retrieve leaf index for given cumulative sum value.
        
        Args:
            idx: Current tree index
            s: Target cumulative sum value
            
        Returns:
            Leaf index corresponding to the sum value
        """
        left = 2 * idx + 1
        right = left + 1
        
        # If we've reached a leaf node
        if left >= len(self.tree):
            return idx
        
        # Navigate left or right based on cumulative sum
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """Get total sum of all priorities."""
        return self.tree[0]
    
    def add(self, priority: float, data: Any):
        """
        Add new experience with given priority.
        
        Args:
            priority: Priority value for the experience
            data: Experience data to store
        """
        idx = self.write_idx + self.capacity - 1  # Convert to tree index
        
        self.data[self.write_idx] = data
        self.update(idx, priority)
        
        self.write_idx = (self.write_idx + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx: int, priority: float):
        """
        Update priority at given tree index.
        
        Args:
            idx: Tree index to update
            priority: New priority value
        """
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, Any]:
        """
        Get experience corresponding to cumulative sum value.

        Args:
            s: Target cumulative sum value

        Returns:
            Tuple of (tree_idx, priority, data)
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1

        # Ensure data_idx is valid
        if data_idx < 0 or data_idx >= self.n_entries:
            # Fallback to a valid index if calculation is wrong
            data_idx = data_idx % self.n_entries if self.n_entries > 0 else 0

        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer for reinforcement learning.
    
    Implements the PER algorithm from Schaul et al. (2015) which samples
    experiences based on their TD error magnitude, allowing the agent to
    learn more efficiently from surprising transitions.
    
    Key features:
    - Priority-based sampling using Sum Tree data structure
    - Importance sampling weights to correct for sampling bias
    - Support for multi-environment training
    
    Reference: https://arxiv.org/abs/1511.05952
    """
    
    def __init__(
        self,
        obs_shape: Union[int, tuple],
        action_shape: Union[int, tuple],
        max_size: int = int(1e6),
        n_envs: int = 1,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 1e-6,
        epsilon: float = 1e-6
    ):
        """
        Initialize Prioritized Replay Buffer.
        
        Args:
            obs_shape: Shape of observations (int or tuple)
            action_shape: Shape of actions (int or tuple)  
            max_size: Maximum buffer size
            n_envs: Number of parallel environments
            alpha: Prioritization exponent (0=uniform, 1=full prioritization)
            beta: Importance sampling exponent (annealed to 1.0 during training)
            beta_increment: Amount to increment beta each step
            epsilon: Small constant to ensure non-zero priorities
        """
        self.max_size = max_size
        self.n_envs = n_envs
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # Validate parameters
        assert 0 <= alpha <= 1, f"alpha must be in [0, 1], got {alpha}"
        assert 0 <= beta <= 1, f"beta must be in [0, 1], got {beta}"
        assert epsilon > 0, f"epsilon must be positive, got {epsilon}"
        
        # Handle shape parameters
        if isinstance(obs_shape, int):
            self.obs_shape = (obs_shape,)
        else:
            self.obs_shape = obs_shape
            
        if isinstance(action_shape, int):
            self.action_shape = (action_shape,)
        else:
            self.action_shape = action_shape
        
        # Initialize Sum Tree for priority management
        self.tree = SumTree(max_size)
        
        # Track maximum priority for new experiences
        self.max_priority = 1.0
        
        # Buffer state
        self.size = 0
    
    @classmethod
    def from_env(
        cls,
        env: gym.vector.VectorEnv,
        max_size: int = int(1e6),
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 1e-6,
        epsilon: float = 1e-6
    ) -> 'PrioritizedReplayBuffer':
        """Create PrioritizedReplayBuffer from environment."""
        obs_shape = env.single_observation_space.shape
        action_shape = env.single_action_space.shape
        n_envs = getattr(env, 'num_envs', 1)
        return cls(obs_shape, action_shape, max_size, n_envs, alpha, beta, beta_increment, epsilon)
    
    def add(self, obs: np.ndarray, action: np.ndarray, reward: Union[float, np.ndarray],
            next_obs: np.ndarray, done: Union[bool, np.ndarray], infos: Optional[list] = None):
        """
        Add transition(s) to the buffer with maximum priority.
        
        New experiences are assigned maximum priority to ensure they are
        sampled at least once before their priority is updated based on TD error.
        """
        # Handle multi-env case
        if self.n_envs > 1:
            batch_size = len(obs) if hasattr(obs, '__len__') else 1
            for i in range(batch_size):
                self._add_single(
                    obs[i] if batch_size > 1 else obs,
                    action[i] if batch_size > 1 else action,
                    reward[i] if hasattr(reward, '__len__') else reward,
                    next_obs[i] if batch_size > 1 else next_obs,
                    done[i] if hasattr(done, '__len__') else done
                )
        else:
            self._add_single(obs, action, reward, next_obs, done)
    
    def _add_single(self, obs: np.ndarray, action: np.ndarray, reward: float,
                   next_obs: np.ndarray, done: bool):
        """Add a single transition with maximum priority."""
        # Convert to numpy arrays and ensure proper shapes
        obs = np.asarray(obs, dtype=np.float32).reshape(self.obs_shape)
        action = np.asarray(action, dtype=np.float32).reshape(self.action_shape)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(self.obs_shape)

        # Create experience tuple
        experience = (
            obs,
            action,
            float(reward),
            next_obs,
            float(done)
        )
        
        # Add to tree with maximum priority
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)
        
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Tuple[Batch, np.ndarray, np.ndarray]:
        """
        Sample a batch of experiences based on their priorities.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (batch_dict, indices, is_weights) where:
            - batch_dict: Standard batch format for training
            - indices: Tree indices for priority updates
            - is_weights: Importance sampling weights
        """
        if self.size < batch_size:
            raise ValueError(f"Buffer size ({self.size}) < batch size ({batch_size})")

        # Sample experiences based on priorities
        indices = []
        experiences = []
        priorities = []

        # Divide priority range into segments for stratified sampling
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            # Sample uniformly within each segment
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)

            # Get experience corresponding to this priority sum
            idx, priority, experience = self.tree.get(s)
            indices.append(idx)
            experiences.append(experience)
            priorities.append(priority)

        # Calculate importance sampling weights with numerical stability
        total_priority = self.tree.total()
        if total_priority <= 0:
            # Fallback: uniform weights if no priorities
            is_weights = np.ones(batch_size, dtype=np.float32)
        else:
            sampling_probs = np.array(priorities) / total_priority
            # Add small epsilon to prevent division by zero
            sampling_probs = np.maximum(sampling_probs, 1e-8)
            is_weights = np.power(self.size * sampling_probs, -self.beta)

            # Normalize by max weight (with safety check)
            max_weight = is_weights.max()
            if max_weight > 0:
                is_weights /= max_weight
            else:
                is_weights = np.ones_like(is_weights)

        # Anneal beta towards 1.0
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Convert experiences to batch format
        obs_batch = []
        action_batch = []
        reward_batch = []
        next_obs_batch = []
        done_batch = []

        for experience in experiences:
            # Safety check for None experiences
            if experience is None:
                raise ValueError("Sampled experience is None - buffer may not be properly initialized")

            obs, action, reward, next_obs, done = experience
            obs_batch.append(obs)
            action_batch.append(action)
            reward_batch.append(reward)
            next_obs_batch.append(next_obs)
            done_batch.append(done)

        batch = create_batch(
            observations=np.array(obs_batch),
            actions=np.array(action_batch),
            rewards=np.array(reward_batch),
            dones=np.array(done_batch),
            next_observations=np.array(next_obs_batch)
        )

        return batch, np.array(indices), is_weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Update priorities for sampled experiences based on TD errors.

        This should be called after computing TD errors for the sampled batch.
        Higher TD errors result in higher sampling priorities.

        Args:
            indices: Tree indices returned from sample()
            priorities: New priority values (typically absolute TD errors)
        """

        for idx, priority in zip(indices, priorities):
            # Ensure non-zero priority and apply alpha exponent
            priority = float(priority)  # Convert to Python float
            priority = max(priority, self.epsilon)
            priority = priority ** self.alpha

            # Update tree and track maximum priority
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return self.size >= batch_size

    def reset(self):
        """Reset the buffer to empty state."""
        self.tree = SumTree(self.max_size)
        self.size = 0
        self.max_priority = 1.0
        self.beta = 0.4  # Reset beta to initial value

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (f"PrioritizedReplayBuffer(size={self.size}/{self.max_size}, "
                f"obs_shape={self.obs_shape}, alpha={self.alpha}, beta={self.beta:.3f})")
