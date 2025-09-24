"""
Observation normalization tool, fully based on SimBA official implementation
Reference: https://github.com/SonyResearch/simba/blob/main/scale_rl/agents/wrappers/utils.py
"""

import numpy as np
from typing import Tuple
import jax.numpy as jnp
from typing import Optional

class RunningMeanStd:
    """
    Track the mean, variance, and count of numbers
    Fully based on SimBA official implementation, using Gymnasium standard algorithm
    """
    
    def __init__(self, epsilon: float = 1e-4, shape: Tuple = (), dtype=np.float32):
        """
        Initialize running time statistics
        
        Args:
            epsilon: numerical stability parameter, SimBA standard is 1e-4
            shape: data shape
            dtype: data type, SimBA uses float32
        """
        self.mean = np.zeros(shape, dtype=dtype)
        self.var = np.ones(shape, dtype=dtype)
        self.count = epsilon  # 🔥 SimBA standard: initial count=epsilon
        self.epsilon = epsilon

    def update(self, x: np.ndarray) -> None:
        """
        Update the mean, variance, and count from batch samples
        
        Args:
            x: input batch data, shape is (batch_size, ...)
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        """
        Update the statistics from the mean, variance, and count of batch samples
        Using SimBA official Welford algorithm implementation
        """
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    def normalize(self, x: np.ndarray, update: bool = True) -> np.ndarray:
        """
        Normalize input data
        
        Args:
            x: input data
            update: update statistics during training, do not update during inference
        Returns:
            normalized data
        """
        if update:
            self.update(x)
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)


def update_mean_var_count_from_moments(
    mean: np.ndarray, 
    var: np.ndarray, 
    count: float, 
    batch_mean: np.ndarray, 
    batch_var: np.ndarray, 
    batch_count: int
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Update the mean, variance, and count from batch samples
    Fully based on SimBA official Welford algorithm implementation
    """
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class ObservationNormalizerWrapper:
    """
    Observation normalization wrapper, fully based on SimBA official implementation
    Reference: https://github.com/SonyResearch/simba/blob/main/scale_rl/agents/wrappers/normalization.py
    """

    def __init__(self, learner, observation_shape: Tuple, epsilon: float = 1e-4):
        """
        Initialize observation normalization wrapper
        
        Args:
            learner: learner to be wrapped
            observation_shape: observation shape
            epsilon: numerical stability parameter
        """
        self.learner = learner
        
        # Using SimBA standard parameters
        self.obs_rms = RunningMeanStd(
            epsilon=epsilon, 
            shape=observation_shape,
            dtype=np.float32
        )
        self.epsilon = epsilon

    def get_action(self, observations: np.ndarray, training: bool = True) -> np.ndarray:
        """
        
        Args:
            observations: environment observations
            training: training mode
            
        Returns:
            sampled actions
        """
        
        # normalize observations
        normalized_obs = self.obs_rms.normalize(observations, update=training)
        
        # call underlying learner (pure network forward pass)
        return self.learner.get_action(normalized_obs)

    def update(self, batch: dict, weights: Optional[jnp.ndarray] = None, training: bool = True) -> dict:
        """
        Update network, automatically handle observation normalization in batch
        Fully based on SimBA official update logic

        Args:
            batch: training batch data
            weights: importance sampling weights
            training: training mode
        Returns:
            training information
        """


        batch = batch.copy()  # do not modify original batch
        batch["observations"] = self.obs_rms.normalize(batch["observations"], update=training)
        batch["next_observations"] = self.obs_rms.normalize(batch["next_observations"], update=training)


        return self.learner.update(batch, weights)

    def __getattr__(self, name):
        """delegate other methods to underlying learner"""
        return getattr(self.learner, name)

    def get_stats(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """get normalized statistics"""
        return self.obs_rms.mean, self.obs_rms.var, self.obs_rms.count

    def set_stats(self, mean: np.ndarray, var: np.ndarray, count: float) -> None:
        """set normalized statistics (for model loading)"""
        self.obs_rms.mean = mean
        self.obs_rms.var = var
        self.obs_rms.count = count