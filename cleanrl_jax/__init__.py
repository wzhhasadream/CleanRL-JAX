"""
JAX-RL: High-Performance Reinforcement Learning in JAX

A modern, GPU-accelerated reinforcement learning library built with JAX and Flax NNX.
This library focuses on sample efficiency, computational speed, and clean, maintainable code.

Key Features:
- GPU-accelerated training with JAX JIT compilation
- Modern neural network API with Flax NNX
- Sample-efficient off-policy algorithms (SAC, TD3)
- Vectorized environments for parallel data collection
- Professional code quality with comprehensive documentation

Algorithms:
- Soft Actor-Critic (SAC): Maximum entropy RL with automatic entropy tuning
- Twin Delayed DDPG (TD3): Deterministic policy gradient with target policy smoothing

Usage:
    >>> from jaxrl_v2.agents.sac import SACLearner
    >>> from jaxrl_v2.agents.td3 import TD3Learner
    >>> import gymnasium as gym
    >>>
    >>> env = gym.vector.SyncVectorEnv([lambda: gym.make("Hopper-v4")])
    >>> sac_learner = SACLearner(env=env, seed=42)

Author: Your Name
License: MIT
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__license__ = "MIT"

# Import main components for easy access
try:
    from .agents.sac.sac_leaner import SACLearner
    from .agents.td3.td3_leaner import TD3Learner
    from .utils.ReplayBuffer import ReplayBuffer

    __all__ = [
        "SACLearner",
        "TD3Learner",
        "ReplayBuffer",
        "__version__",
        "__author__",
        "__email__",
        "__license__",
    ]

except ImportError as e:
    # Handle import errors gracefully during package installation
    import warnings
    warnings.warn(f"Some components could not be imported: {e}")

    __all__ = [
        "__version__",
        "__author__",
        "__email__",
        "__license__",
    ]
