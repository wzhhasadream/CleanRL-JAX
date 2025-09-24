import jax
import jax.numpy as jnp
from typing import Dict, Any



class GradientMonitor:
    """Gradient monitor for tracking gradient health"""
    
    def __init__(self, 
                 monitor_frequency: int = 100,
                 sparsity_threshold: float = 1e-8):
        """
        Args:
            monitor_frequency: Monitoring frequency (how often to monitor)
            sparsity_threshold: Threshold for determining if a gradient is zero
        """
        self.monitor_frequency = monitor_frequency
        self.sparsity_threshold = sparsity_threshold
        self.step_count = 0
        
    
    def _flatten_gradients(self, grads: Any) -> jnp.ndarray:
        """Flatten the gradient tree into a 1D array"""
        flat_grads = []
        for grad in jax.tree_util.tree_leaves(grads):
            if grad is not None:  
                flat_grads.append(grad.flatten())
        
        if flat_grads:
            return jnp.concatenate(flat_grads)
        else:
            return jnp.array([])
    
    def _compute_gradient_stats(self, flat_grads: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Compute gradient statistics"""
        # Compute various norms
        l1_norm = jnp.sum(jnp.abs(flat_grads))
        l2_norm = jnp.sqrt(jnp.sum(flat_grads ** 2))

        # Compute sparsity (proportion of gradients close to zero)
        near_zero_count = jnp.sum(jnp.abs(flat_grads) < self.sparsity_threshold)
        sparsity = near_zero_count / flat_grads.shape[0]

        return {
            'grad_l1_norm': l1_norm,
            'grad_l2_norm': l2_norm,
            'GraMa': sparsity,   # Reference:https://arxiv.org/abs/2505.24061
        }
    
    
    def should_monitor(self) -> bool:
        """Determine if monitoring should be performed"""
        self.step_count += 1
        return self.step_count % self.monitor_frequency == 0
    
    def monitor_gradients(self, grads: Any) -> Dict[str, float]:
        """
        Monitor gradient statistics
        
        Args:
            grads: Gradient PyTree
            
        Returns:
            Dictionary containing gradient statistics
        """
        if not self.should_monitor():
            return {}
    
        # Flatten all gradients
        flat_grads = self._flatten_gradients(grads)

        if len(flat_grads) == 0:
            return {}

        # Compute global statistics
        raw_stats = self._compute_gradient_stats(flat_grads)

        # Convert to Python float type
        stats = {k: float(v) for k, v in raw_stats.items()}

        return stats