import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
from flax import nnx
from typing import Optional, Tuple
from tensorflow_probability.substrates import jax as tfp
from cleanrl_jax.utils.SimBa import SimBaEncoder

# TensorFlow Probability distributions and bijectors
tfd = tfp.distributions
tfb = tfp.bijectors

########################
# Network Constants
########################

# Log standard deviation bounds for policy network
# These bounds prevent the policy from becoming too deterministic or too stochastic
LOG_STD_MIN = -5  # Minimum log std (std ≈ 0.007, quite deterministic)
LOG_STD_MAX = 2   # Maximum log std (std ≈ 7.4, quite stochastic)



def default_init(scale: Optional[float] = jnp.sqrt(2)):
    """
    Default weight initialization for neural networks.

    Uses orthogonal initialization which helps with gradient flow and training stability.
    The scale factor √2 is commonly used for ReLU activations.

    Args:
        scale: Scaling factor for the orthogonal initialization

    Returns:
        Orthogonal initializer function
    """
    return nnx.initializers.orthogonal(scale)


########################################################
# Critic Networks (Q-functions)
########################################################


class SoftQNetwork(nnx.Module):
    def __init__(
        self,
        env: gym.vector.VectorEnv,
        rngs: nnx.Rngs,
        hidden_dim: int = 512,
        num_blocks: int = 2,
        block_type: str = "residual"
    ):

        self.obs_dim = env.single_observation_space.shape[0]
        self.action_dim = env.single_action_space.shape[0]
        self.hidden_dim = hidden_dim

        self.encoder = SimBaEncoder(
            input_dim=self.obs_dim + self.action_dim,
            hidden_dim=hidden_dim,
            rngs=rngs,
            block_type=block_type,
            num_blocks=num_blocks
        )
        self.fc = nnx.Linear(hidden_dim, 1, rngs=rngs, kernel_init=default_init())


    def __call__(self, x: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([x, a], axis=1)
        x = self.encoder(x)
        x = self.fc(x)
        return x
    

class DoubleCritic(nnx.Module):
    """Double soft Q network using NNX API."""

    def __init__(
        self,
        env: gym.vector.VectorEnv,
        rngs: nnx.Rngs = None,
        hidden_dim: int = 512,
        num_blocks: int = 2,
        block_type: str = "residual"
    ):
        self.critic1 = SoftQNetwork(
            env, rngs=rngs,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            block_type=block_type
        )
        self.critic2 = SoftQNetwork(
            env, rngs=rngs,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            block_type=block_type
        )

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        q1 = self.critic1(observations, actions)
        q2 = self.critic2(observations, actions)
        return jnp.minimum(q1, q2)

    def get_both_q_values(self, observations: jnp.ndarray, actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        q1 = self.critic1(observations, actions)
        q2 = self.critic2(observations, actions)
        return q1, q2
    
########################################################
# Actor Network
########################################################``

class Actor(nnx.Module):
    def __init__(
        self,
        env: gym.vector.VectorEnv,
        rngs: nnx.Rngs,
        hidden_dim: int = 128,
        num_blocks: int = 1,
        block_type: str = "residual"
    ):

        self.obs_dim = env.single_observation_space.shape[0]
        self.action_dim = env.single_action_space.shape[0]
        self.hidden_dim = hidden_dim

        self.rngs = rngs

        self.encoder = SimBaEncoder(
            input_dim=self.obs_dim,
            hidden_dim=hidden_dim,
            rngs=rngs,
            block_type=block_type,
            num_blocks=num_blocks
        )
        self.fc_mean = nnx.Linear(hidden_dim, self.action_dim, rngs=rngs, kernel_init=default_init())
        self.fc_logstd = nnx.Linear(hidden_dim, self.action_dim, rngs=rngs, kernel_init=default_init())

        action_high = jnp.array(env.single_action_space.high, dtype=jnp.float32)
        action_low = jnp.array(env.single_action_space.low, dtype=jnp.float32)
        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias = (action_high + action_low) / 2.0

    def __call__(self, x: jnp.ndarray) -> tfd.TransformedDistribution:
        """
        Forward pass through pure SimBA architecture.

        Args:
            x: Input observations (already normalized by wrapper)
        """
        x = self.encoder(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)

        log_std = jnp.tanh(log_std)

        # Scale log_std to [LOG_STD_MIN, LOG_STD_MAX]
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        # Convert log_std to std (must be positive for scale_diag)
        std = jnp.exp(log_std)

        base_dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=std)

        bijector = tfb.Chain(
            [tfb.Shift(shift=self.action_bias), 
            tfb.Scale(scale=self.action_scale), 
            tfb.Tanh()]
            )

        action_distribution = tfd.TransformedDistribution(base_dist, bijector)

        return action_distribution
    

    def get_action(self, x: jnp.ndarray, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
        action_distribution = self(x)
        action = action_distribution.sample(seed=key)
        log_prob = action_distribution.log_prob(action)


        return action, log_prob


class Alpha(nnx.Module):
    def __init__(self, init_value: float = 0.0):
        # ensure alpha is positive by storing log_alpha
        self.log_alpha = nnx.Param(jnp.array(init_value))

    def __call__(self) -> jnp.ndarray:
        return jnp.exp(self.log_alpha.value)  