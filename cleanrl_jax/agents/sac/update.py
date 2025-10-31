"""
SAC update functions using Flax NNX API.
Based on: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
"""

import jax
import jax.numpy as jnp
from flax import nnx
from typing import Dict, Tuple, Union, Optional
from .network import Actor, DoubleCritic, Alpha

import numpy as np

@nnx.jit
def get_action(actor: Actor, observations: np.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
    action, _ = actor.get_action(observations, key)
    return action


def update_critic(
    critic: DoubleCritic,
    target_critic: DoubleCritic,
    actor: Actor,
    batch: Dict[str, jnp.ndarray],
    alpha_value: float,
    gamma: float,
    critic_optimizer: nnx.Optimizer,
    key: jax.random.PRNGKey
) -> Tuple[jnp.ndarray, Dict[str, float]]:

    def critic_loss_fn(critic):

        next_actions, next_log_pi = actor.get_action(batch["next_observations"], key)

        next_q1 = target_critic.critic1(batch["next_observations"], next_actions)
        next_q2 = target_critic.critic2(batch["next_observations"], next_actions)

        # Take minimum to reduce overestimation bias (key insight from TD3/SAC)
        min_next_q = jnp.minimum(next_q1, next_q2)

        # SAC Bellman equation: Q_target = r + γ(1-done)(Q_min - α*log_π)
        # The entropy term (α*log_π) encourages exploration
        target_q = batch["rewards"] + (1 - batch["dones"]) * gamma * (min_next_q - alpha_value * next_log_pi)


        target_q = jax.lax.stop_gradient(target_q)

        q1 = critic.critic1(batch["observations"], batch["actions"])
        q2 = critic.critic2(batch["observations"], batch["actions"])

        q1_loss = jnp.mean((q1 - target_q) ** 2) 
        q2_loss = jnp.mean((q2 - target_q) ** 2) 
        critic_loss = q1_loss + q2_loss

        # Info for logging
        info = {
            'q1_mean': jnp.mean(q1),
            'q2_mean': jnp.mean(q2),
            'q1_loss': q1_loss,
            'q2_loss': q2_loss
        }

        return critic_loss, info

    # Compute gradients and update
    (loss, info), grads = nnx.value_and_grad(critic_loss_fn, has_aux=True)(critic)

    info.update({"critic_grad":grads})

    critic_optimizer.update(grads)



    return loss, info


def update_critic_with_weights(
    critic: DoubleCritic,
    target_critic: DoubleCritic,
    actor: Actor,
    batch: Dict[str, jnp.ndarray],
    alpha_value: float,
    gamma: float,
    critic_optimizer: nnx.Optimizer,
    key: jax.random.PRNGKey,
    weights: jnp.ndarray
) -> Tuple[jnp.ndarray, Dict[str, float]]:
    """
    weights: importance sampling weights (for PER)
    """
    def critic_loss_fn(critic):

        next_actions, next_log_pi = actor.get_action(batch["next_observations"], key)

        next_q1 = target_critic.critic1(batch["next_observations"], next_actions)
        next_q2 = target_critic.critic2(batch["next_observations"], next_actions)

        min_next_q = jnp.minimum(next_q1, next_q2)

        # SAC Bellman equation: Q_target = r + γ(1-done)(Q_min - α*log_π)
        # The entropy term (α*log_π) encourages exploration
        target_q = batch["rewards"] + (1 - batch["dones"]) * gamma * (min_next_q - alpha_value * next_log_pi)


        target_q = jax.lax.stop_gradient(target_q)

        q1 = critic.critic1(batch["observations"], batch["actions"])
        q2 = critic.critic2(batch["observations"], batch["actions"])

        q1_loss = jnp.mean((q1 - target_q) ** 2 * weights) # importance sampling weights
        q2_loss = jnp.mean((q2 - target_q) ** 2 * weights) # importance sampling weights
        critic_loss = q1_loss + q2_loss

        # Info for logging
        info = {
            'q1_mean': jnp.mean(q1),
            'q2_mean': jnp.mean(q2),
            'q1_loss': q1_loss,
            'q2_loss': q2_loss
        }

        return critic_loss, info

    # Compute gradients and update
    (loss, info), grads = nnx.value_and_grad(critic_loss_fn, has_aux=True)(critic)

    info.update({"critic_grad":grads})

    critic_optimizer.update(grads)



    return loss, info

def update_actor(
    actor: Actor,
    critic: DoubleCritic,
    batch: Dict[str, jnp.ndarray],
    alpha_value: float,
    actor_optimizer: nnx.Optimizer,
    key: jax.random.PRNGKey
) -> Tuple[jnp.ndarray, Dict[str, float]]:
    def actor_loss_fn(actor):
        actions, log_pi = actor.get_action(batch["observations"], key)
        q1 = critic.critic1(batch["observations"], actions)
        q2 = critic.critic2(batch["observations"], actions)
        min_q = jnp.minimum(q1, q2)
        actor_loss = -jnp.mean(min_q - alpha_value * log_pi)
        return actor_loss, {"actor_loss": actor_loss}

    (loss, info), grads = nnx.value_and_grad(actor_loss_fn, has_aux=True)(actor)


    actor_optimizer.update(grads)

    return loss, info


def update_alpha(
    alpha: Alpha,
    actor: Actor,
    batch: Dict[str, jnp.ndarray],
    key: jax.random.PRNGKey,
    target_entropy: float,
    alpha_optimizer: nnx.Optimizer
) -> Tuple[jnp.ndarray, Dict[str, float]]:

    def alpha_loss_fn(alpha):
        _, log_pi = actor.get_action(batch["observations"], key)
        alpha_loss = (-alpha() * (log_pi + target_entropy)).mean()  
        return alpha_loss, {"alpha_loss": alpha_loss, "alpha_value": alpha()}
    
    (loss, info), grads = nnx.value_and_grad(alpha_loss_fn, has_aux=True)(alpha)


    alpha_optimizer.update(grads)

    return loss, info



def update_target_networks(target_critic: DoubleCritic, critic: DoubleCritic, tau: float):
    """Soft update target networks."""
    online_params = nnx.state(critic)
    target_params = nnx.state(target_critic)

    # Soft update: target = tau * online + (1 - tau) * target
    new_params = jax.tree.map(
        lambda online, target: tau * online + (1 - tau) * target,
        online_params, target_params
    )

    nnx.update(target_critic, new_params)


@nnx.jit(static_argnames=('autotune','update_policy','update_target_network','policy_frequency'))
def update_sac(
    actor: Actor,
    critic: DoubleCritic,
    target_critic: DoubleCritic,
    alpha: Union[Alpha, float],
    batch: Dict[str, jnp.ndarray],
    key: jax.random.PRNGKey,
    target_entropy: float,
    alpha_optimizer: nnx.Optimizer,
    gamma: float,
    tau: float,
    autotune: bool,
    policy_optimizer: nnx.Optimizer,
    critic_optimizer: nnx.Optimizer,
    update_policy: bool,
    update_target_network: bool,
    policy_frequency: int
) -> Dict[str, float]:

    critic_key, policy_key = jax.random.split(key, 2)
    info = {}

    if autotune:
        alpha_value = alpha()
    else:
        alpha_value = alpha

    # always update critic
    q_loss, critic_info = update_critic(
        critic, target_critic, actor, batch,
        alpha_value, gamma, critic_optimizer, critic_key
    )
    info.update({"q_loss": q_loss, **critic_info})

    if update_policy:
        for i in range(policy_frequency):
            iter_actor_key, iter_alpha_key, policy_key = jax.random.split(
                policy_key, 3
            )
            
            actor_loss, policy_info = update_actor(
                actor, critic, batch, alpha_value,
                policy_optimizer, iter_actor_key
            )

            if autotune:
                alpha_loss, alpha_info = update_alpha(
                    alpha, actor, batch, iter_alpha_key,
                    target_entropy, alpha_optimizer
                )
                info.update({"alpha_loss": alpha_loss, **alpha_info})

        info.update({"actor_loss": actor_loss, **policy_info})
        

    if update_target_network:
        update_target_networks(target_critic, critic, tau)

    info.update({"alpha_value": alpha_value})
    return info


@nnx.jit(static_argnames=('autotune','update_policy','update_target_network','policy_frequency'))
def update_sac_with_weights(
    actor: Actor,
    critic: DoubleCritic,
    target_critic: DoubleCritic,
    alpha: Union[Alpha, float],
    batch: Dict[str, jnp.ndarray],
    key: jax.random.PRNGKey,
    weights: jnp.ndarray,
    target_entropy: float,
    alpha_optimizer: nnx.Optimizer,
    gamma: float,
    tau: float,
    autotune: bool,
    policy_optimizer: nnx.Optimizer,
    critic_optimizer: nnx.Optimizer,
    update_policy: bool,
    update_target_network: bool,
    policy_frequency: int
) -> Dict[str, float]:

    critic_key, policy_key = jax.random.split(key, 2)
    info = {}

    if autotune:
        alpha_value = alpha()
    else:
        alpha_value = alpha

    # always update critic
    q_loss, critic_info = update_critic_with_weights(
        critic, target_critic, actor, batch,
        alpha_value, gamma, critic_optimizer, critic_key, weights
    )
    info.update({"q_loss": q_loss, **critic_info})

    if update_policy:
        for i in range(policy_frequency):
            iter_actor_key, iter_alpha_key, policy_key = jax.random.split(
                policy_key, 3
            )
            
            actor_loss, policy_info = update_actor(
                actor, critic, batch, alpha_value,
                policy_optimizer, iter_actor_key
            )

            if autotune:
                alpha_loss, alpha_info = update_alpha(
                    alpha, actor, batch, iter_alpha_key,
                    target_entropy, alpha_optimizer
                )
                info.update({"alpha_loss": alpha_loss, **alpha_info})

        info.update({"actor_loss": actor_loss, **policy_info})
        

    if update_target_network:
        update_target_networks(target_critic, critic, tau)

    info.update({"alpha_value": alpha_value})
    return info



