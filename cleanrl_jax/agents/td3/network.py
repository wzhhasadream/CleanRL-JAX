import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
from flax import nnx
from typing import Optional, Tuple
from cleanrl_jax.utils.SimBa import SimBaEncoder



def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nnx.initializers.orthogonal(scale)

########################
# Critic Network
########################


class QNetwork(nnx.Module):
    def __init__(self,
                env: gym.vector.VectorEnv, 
                rngs: nnx.Rngs,
                hidden_dim: int = 512,
                num_blocks: int = 2,
                block_type: str = "residual"
    ):

        self.obs_dim = env.single_observation_space.shape[0]
        self.action_dim = env.single_action_space.shape[0]

        self.encoder = SimBaEncoder(self.obs_dim + self.action_dim, hidden_dim, rngs=rngs, block_type=block_type,num_blocks=num_blocks)
        self.fc = nnx.Linear(hidden_dim, 1, rngs=rngs, kernel_init=default_init())


    def __call__(self, x: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([x, a], axis=1)
        x = self.encoder(x)
        x = self.fc(x)
        return x

class DoubleCritic(nnx.Module):

    def __init__(
        self,
        env: gym.vector.VectorEnv,
        rngs: nnx.Rngs,
        hidden_dim: int = 512,
        num_blocks: int = 2,
        block_type: str = "residual"
    ):
        self.critic1 = QNetwork(env, rngs=rngs, hidden_dim=hidden_dim, num_blocks=num_blocks, block_type=block_type)
        self.critic2 = QNetwork(env, rngs=rngs, hidden_dim=hidden_dim, num_blocks=num_blocks, block_type=block_type)

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        q1 = self.critic1(observations, actions)
        q2 = self.critic2(observations, actions)
        return jnp.minimum(q1, q2)

    def get_both_q_values(self, observations: jnp.ndarray, actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        q1 = self.critic1(observations, actions)
        q2 = self.critic2(observations, actions)
        return q1, q2

########################
# Actor Network
########################

class Actor(nnx.Module):
    def __init__(self, env: gym.vector.VectorEnv, rngs: nnx.Rngs, hidden_dim: int = 128, num_blocks: int = 1, block_type: str = "residual"):
        self.obs_dim = env.single_observation_space.shape[0]
        self.action_dim = env.single_action_space.shape[0]
        self.action_low = env.single_action_space.low
        self.action_high = env.single_action_space.high

        self.encoder = SimBaEncoder(self.obs_dim, hidden_dim, rngs=rngs, block_type=block_type,num_blocks=num_blocks)
        self.fc = nnx.Linear(hidden_dim, self.action_dim, rngs=rngs, kernel_init=default_init())

        # scale and bias calculation
        self.action_bias = (env.single_action_space.high + env.single_action_space.low) / 2  # center point
        self.action_scale = (env.single_action_space.high - env.single_action_space.low) / 2  # scaling factor

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.encoder(x)
        x = self.fc(x)
        x = jnp.tanh(x) * self.action_scale + self.action_bias
        return x