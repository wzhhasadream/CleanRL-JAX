"""
SAC Learner implementation using Flax NNX API.
Based on: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
"""



from typing import Dict, Optional
import numpy as np
import gymnasium as gym

from flax import nnx
import jax
import jax.numpy as jnp
import optax

from .network import Actor, DoubleCritic, Alpha
from .update import update_sac, get_action,update_sac_with_weights




class SACLearner:
    """
    Soft Actor-Critic (SAC) Learner

    This class implements the complete SAC algorithm including:
    - Actor network (policy) with stochastic actions
    - Double critic networks (Q-functions)
    - Automatic entropy tuning (temperature parameter alpha)
    - Target networks with soft updates
    - Experience replay and off-policy learning

    The implementation follows the SAC algorithm from Haarnoja et al. (2019)
    with automatic entropy tuning for optimal exploration-exploitation balance.
    """

    def __init__(
        self,
        env: gym.vector.VectorEnv,
        seed: int = 1,
        policy_lr: float = 1e-4,
        q_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_frequency: int = 2,
        target_network_frequency: int = 1,
        alpha: float = 0.2,
        autotune: bool = True,

        # SimBA network architecture parameters
        actor_hidden_dim: int = 128,
        actor_num_blocks: int = 1,
        actor_block_type: str = "residual",
        critic_hidden_dim: int = 512,
        critic_num_blocks: int = 2,
        critic_block_type: str = "residual",

    ):
        """
        Initialize the SAC learner with networks and optimizers.

        Args:
            env: Vectorized gymnasium environment
            seed: Random seed for reproducibility
            policy_lr: Learning rate for the actor (policy) network
            q_lr: Learning rate for the critic (Q-function) networks
            gamma: Discount factor for future rewards (0 < gamma <= 1)
            tau: Soft update coefficient for target networks (0 < tau <= 1)
            policy_frequency: Frequency of policy updates (every N critic updates)
            target_network_frequency: Frequency of target network updates
            alpha: Initial entropy regularization coefficient (if not autotuning)
            autotune: Whether to automatically tune the entropy coefficient alpha
            actor_hidden_dim: hidden dimension for actor
            actor_num_blocks: number of blocks for actor
            actor_block_type: type of block for actor
            critic_hidden_dim: hidden dimension for critic
            critic_num_blocks: number of blocks for critic
            critic_block_type: type of block for critic
        """
        # Store hyperparameters
        self.gamma = gamma  # Discount factor for future rewards
        self.tau = tau  # Soft update coefficient for target networks
        self.policy_frequency = policy_frequency  # Policy update frequency (delayed updates)
        self.target_network_frequency = target_network_frequency  # Target network update frequency
        self.autotune = autotune  # Whether to automatically tune entropy coefficient



        self.prng = jax.random.PRNGKey(seed)


        actor_rngs = nnx.Rngs(seed)
        qf_rngs = nnx.Rngs(seed + 1)  
        qf_target_rngs = nnx.Rngs(seed + 2)


        self.actor = Actor(
            env, actor_rngs,
            hidden_dim=actor_hidden_dim,
            num_blocks=actor_num_blocks,
            block_type=actor_block_type
        )


        self.qf = DoubleCritic(
            env, qf_rngs,
            hidden_dim=critic_hidden_dim,
            num_blocks=critic_num_blocks,
            block_type=critic_block_type
        )


        self.qf_target = DoubleCritic(
            env, qf_target_rngs,
            hidden_dim=critic_hidden_dim,
            num_blocks=critic_num_blocks,
            block_type=critic_block_type
        )


        self.alpha = Alpha() if autotune else alpha

        if autotune:
            self.a_optimizer = nnx.Optimizer(self.alpha, optax.adamw(q_lr))
        else:
            self.a_optimizer = None


        nnx.update(self.qf_target, nnx.state(self.qf))


        self.qf_optimizer = nnx.Optimizer(self.qf, optax.adamw(q_lr,weight_decay=1e-2))  # Critic optimizer
        self.actor_optimizer = nnx.Optimizer(self.actor, optax.adamw(policy_lr,weight_decay=1e-2))  # Actor optimizer


        self.target_entropy = -env.single_action_space.shape[0]


        self.global_step = 0


    def get_action(self, observations: np.ndarray) -> np.ndarray:


        self.prng, key = jax.random.split(self.prng)

        action = get_action(self.actor, observations, key)

        return np.array(action)


    
    def update(self, batch: Dict[str, jnp.ndarray], weights: Optional[jnp.ndarray] = None) -> Dict[str, float]:
        """
        Update SAC networks with optional importance sampling weights.

        Args:
            batch: Training batch data
            weights: Importance sampling weights for PER (optional)
        """
        self.global_step += 1
        self.prng, key = jax.random.split(self.prng)

        # Use uniform weights if no importance sampling weights provided
        if weights is None:
            info = update_sac(
                self.actor, self.qf, self.qf_target, self.alpha, batch,
                key, self.target_entropy, self.a_optimizer, self.gamma,
                self.tau, self.autotune, self.actor_optimizer, self.qf_optimizer,
                self.global_step % self.policy_frequency == 0,
                self.global_step % self.target_network_frequency == 0,
                self.policy_frequency
            )
        else:
            info = update_sac_with_weights(
                self.actor, self.qf, self.qf_target, self.alpha, batch,
                key, weights, self.target_entropy, self.a_optimizer, self.gamma,
                self.tau, self.autotune, self.actor_optimizer, self.qf_optimizer,
                self.global_step % self.policy_frequency == 0,
                self.global_step % self.target_network_frequency == 0,
                self.policy_frequency
            )

        return info



