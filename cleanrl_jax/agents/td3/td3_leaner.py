"""
TD3 Learner implementation using Flax NNX API.
Based on: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py
"""




from .network import Actor, DoubleCritic
from .update import get_action, update_td3, update_td3_with_weights
import gymnasium as gym
from flax import nnx
import optax
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Optional


class TD3Learner:
    def __init__(self,
    env: gym.vector.VectorEnv,
    gamma: float = 0.99,
    tau: float = 0.005,
    lr: float = 3e-4,
    policy_noise: float = 0.2,
    noise_clip: float = 0.5,
    policy_frequency: int = 2,
    seed: int = 1,
    critic_hidden_dim: int = 512,
    critic_num_blocks: int = 2,
    critic_block_type: str = "residual",
    actor_hidden_dim: int = 128,
    actor_num_blocks: int = 1,
    actor_block_type: str = "residual"
):
        self.actor = Actor(env,nnx.Rngs(seed), hidden_dim=actor_hidden_dim, num_blocks=actor_num_blocks, block_type=actor_block_type)
        self.critic = DoubleCritic(env,nnx.Rngs(seed+1), hidden_dim=critic_hidden_dim, num_blocks=critic_num_blocks, block_type=critic_block_type)
        self.target_critic = DoubleCritic(env,nnx.Rngs(seed+2), hidden_dim=critic_hidden_dim, num_blocks=critic_num_blocks, block_type=critic_block_type)
        self.target_actor = Actor(env,nnx.Rngs(seed+3), hidden_dim=actor_hidden_dim, num_blocks=actor_num_blocks, block_type=actor_block_type)
        nnx.update(self.target_critic, nnx.state(self.critic))
        nnx.update(self.target_actor, nnx.state(self.actor))

        self.actor_optimizer = nnx.Optimizer(self.actor, optax.adam(lr))
        self.critic_optimizer = nnx.Optimizer(self.critic, optax.adam(lr))

        self.gamma = gamma
        self.tau = tau
        self.policy_frequency = policy_frequency
        self.seed = seed

        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

        self.global_step = 0
        self.key = jax.random.PRNGKey(self.seed)

    def update(self, batch: Dict[str, jnp.ndarray], weights: Optional[jnp.ndarray] = None) -> Dict[str, float]:
        self.key, key = jax.random.split(self.key)
        if weights is None:
            info = update_td3(
                actor=self.actor,
                critic=self.critic,
                target_actor=self.target_actor,
                target_critic=self.target_critic,
                batch=batch,
                gamma=self.gamma,
                tau=self.tau,
                policy_noise=self.policy_noise,
                noise_clip=self.noise_clip,
                actor_optimizer=self.actor_optimizer,
                critic_optimizer=self.critic_optimizer,
                key=key,
                update_policy=self.global_step % self.policy_frequency == 0
            )
        else:
            info = update_td3_with_weights(
                actor=self.actor,
                critic=self.critic,
                target_actor=self.target_actor,
                target_critic=self.target_critic,
                weights=weights,
                batch=batch,
                gamma=self.gamma,
                tau=self.tau,
                policy_noise=self.policy_noise,
                noise_clip=self.noise_clip,
                actor_optimizer=self.actor_optimizer,
                critic_optimizer=self.critic_optimizer,
                key=key,
                update_policy=self.global_step % self.policy_frequency == 0
            )
        self.global_step += 1
        return info


    def get_action(self, observations: jnp.ndarray) -> np.ndarray:
        return np.array(get_action(self.actor, observations))



