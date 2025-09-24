

from flax import nnx
import jax.numpy as jnp
from typing import Any


def he_normal_init():
    """He normal initialization for neural network weights."""
    return nnx.initializers.he_normal()


def orthogonal_init(scale: float = 1.0):
    """Orthogonal initialization with custom scaling."""
    return nnx.initializers.orthogonal(scale)


class MLPBlock(nnx.Module):
    """
    Standard MLP block for comparison with ResidualBlock.
    
    Architecture:
    - Linear(input_dim -> hidden_dim) + ReLU
    - Linear(hidden_dim -> hidden_dim) + ReLU
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dtype: Any = jnp.float32,
        rngs: nnx.Rngs = None
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dtype = dtype

        # Initialize layers with orthogonal weights (sqrt(2) scaling for ReLU)
        self.dense1 = nnx.Linear(
            in_features=input_dim,
            out_features=hidden_dim,
            kernel_init=orthogonal_init(jnp.sqrt(2)),
            dtype=dtype,
            rngs=rngs
        )

        self.dense2 = nnx.Linear(
            in_features=hidden_dim,
            out_features=hidden_dim,
            kernel_init=orthogonal_init(jnp.sqrt(2)),
            dtype=dtype,
            rngs=rngs
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the MLP block."""
        x = self.dense1(x)
        x = nnx.relu(x)
        x = self.dense2(x)
        x = nnx.relu(x)
        return x


class ResidualBlock(nnx.Module):
    """
    Residual block used in SimBa architecture.
    
    Architecture:
    - LayerNorm
    - Linear(hidden_dim -> hidden_dim * 4) + ReLU
    - Linear(hidden_dim * 4 -> hidden_dim)
    - Residual connection
    """
    def __init__(
        self,
        hidden_dim: int,
        dtype: Any = jnp.float32,
        rngs: nnx.Rngs = None
    ):
        self.hidden_dim = hidden_dim
        self.dtype = dtype

        # Layer normalization
        self.layer_norm = nnx.LayerNorm(
            num_features=hidden_dim,
            dtype=dtype,
            rngs=rngs
        )

        # Feedforward network with 4x expansion
        self.dense1 = nnx.Linear(
            in_features=hidden_dim,
            out_features=hidden_dim * 4,
            kernel_init=he_normal_init(),
            dtype=dtype,
            rngs=rngs
        )

        self.dense2 = nnx.Linear(
            in_features=hidden_dim * 4,
            out_features=hidden_dim,
            kernel_init=he_normal_init(),
            dtype=dtype,
            rngs=rngs
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass with residual connection."""
        # Store residual connection
        residual = x

        # Pre-norm residual block
        x = self.layer_norm(x)
        x = self.dense1(x)
        x = nnx.relu(x)
        x = self.dense2(x)

        # Add residual connection
        return residual + x

class SimBaEncoder(nnx.Module):
    """
    SimBa encoder supporting both MLP and residual block architectures.

    Args:
        input_dim: Dimension of input features
        hidden_dim: Dimension of hidden layers
        num_blocks: Number of residual blocks (default: 1)
        block_type: Type of block to use ('residual' or 'mlp')
        dtype: Data type for computations
        rngs: Random number generators for initialization

    Returns:
        jnp.ndarray: Encoded features
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int = 1,
        block_type: str = 'residual',
        dtype: Any = jnp.float32,
        rngs: nnx.Rngs = None
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.block_type = block_type
        self.dtype = dtype

        if block_type == 'mlp':
            self.encoder = MLPBlock(input_dim, hidden_dim, dtype=dtype, rngs=rngs)
        elif block_type == 'residual':
            # Initial projection to hidden_dim
            self.input_projection = nnx.Linear(
                in_features=input_dim,
                out_features=hidden_dim,
                kernel_init=orthogonal_init(1.0),
                dtype=dtype,
                rngs=rngs
            )

            # Stack residual blocks
            self.residual_blocks = [
                ResidualBlock(hidden_dim, dtype=dtype, rngs=rngs)
                for _ in range(num_blocks)
            ]

            # Final layer norm
            self.final_layer_norm = nnx.LayerNorm(
                num_features=hidden_dim,
                dtype=dtype,
                rngs=rngs
            )
        else:
            raise ValueError(f"Unknown block_type: {block_type}")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the encoder.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            encoded: Encoded features of shape (batch_size, hidden_dim)
        """
        if self.block_type == 'mlp':
            return self.encoder(x)
        elif self.block_type == 'residual':
            # Initial projection
            x = self.input_projection(x)

            # Apply residual blocks
            for block in self.residual_blocks:
                x = block(x)

            # Final layer normalization
            x = self.final_layer_norm(x)

            return x