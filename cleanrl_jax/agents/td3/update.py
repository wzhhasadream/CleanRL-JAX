import jax
import jax.numpy as jnp
from flax import nnx
from typing import Dict, Tuple
from flax import nnx
from .network import Actor, DoubleCritic


@nnx.jit
def get_action(
    actor: Actor,
    observations: jnp.ndarray,
) -> jnp.ndarray:
    action = actor(observations)
    return action



def update_critic(
    critic: DoubleCritic,
    target_critic: DoubleCritic,
    target_actor: Actor,
    batch: Dict[str, jnp.ndarray],
    gamma: float,
    policy_noise: float,
    noise_clip: float,
    critic_optimizer: nnx.Optimizer,
    key: jax.random.PRNGKey
) -> Tuple[jnp.ndarray, Dict[str, float]]:
    """
    TD3 Critic update with target policy smoothing.
    """
    def critic_loss_fn(critic):
        # Generate target actions with smoothing noise
        target_actions = target_actor(batch["next_observations"])

        # Add clipped noise for target policy smoothing
        noise = jax.random.normal(key, target_actions.shape) * policy_noise
        clipped_noise = jnp.clip(noise, -noise_clip, noise_clip) * target_actor.action_scale

        # Clip target actions to action bounds
        noisy_target_actions = jnp.clip(
            target_actions + clipped_noise,
            target_actor.action_low,
            target_actor.action_high
        )

        # Compute target Q-values (take minimum of two critics)
        target_q = target_critic(batch["next_observations"], noisy_target_actions)

        # Compute target values
        target_values = batch["rewards"] + gamma * (1 - batch["dones"]) * target_q

        # Compute current Q-values
        q1, q2 = critic.get_both_q_values(batch["observations"], batch["actions"])

        # Compute losses
        q1_loss = jnp.mean((q1 - target_values) ** 2)
        q2_loss = jnp.mean((q2 - target_values) ** 2)
        total_loss = q1_loss + q2_loss

        # Info for logging
        info = {
            'q1_loss': q1_loss,
            'q2_loss': q2_loss,
            'q1_mean': jnp.mean(q1),
            'q2_mean': jnp.mean(q2),
            'target_q_mean': jnp.mean(target_q),
        }

        return total_loss, info

    # Compute gradients and update
    (loss, info), grads = nnx.value_and_grad(critic_loss_fn, has_aux=True)(critic)

    info.update({"critic_grad":grads})
    critic_optimizer.update(grads)

    return loss, info


def update_critic_with_weights(
    critic: DoubleCritic,
    target_critic: DoubleCritic,
    target_actor: Actor,
    batch: Dict[str, jnp.ndarray],
    gamma: float,
    policy_noise: float,
    noise_clip: float,
    critic_optimizer: nnx.Optimizer,
    weights: jnp.ndarray,
    key: jax.random.PRNGKey
) -> Tuple[jnp.ndarray, Dict[str, float]]:
    """
    TD3 Critic update with target policy smoothing.
    """
    def critic_loss_fn(critic):
        # Generate target actions with smoothing noise
        target_actions = target_actor(batch["next_observations"])

        # Add clipped noise for target policy smoothing
        noise = jax.random.normal(key, target_actions.shape) * policy_noise
        clipped_noise = jnp.clip(noise, -noise_clip, noise_clip) * target_actor.action_scale

        # Clip target actions to action bounds
        noisy_target_actions = jnp.clip(
            target_actions + clipped_noise,
            target_actor.action_low,
            target_actor.action_high
        )

        # Compute target Q-values (take minimum of two critics)
        target_q = target_critic(batch["next_observations"], noisy_target_actions)

        # Compute target values
        target_values = batch["rewards"] + gamma * (1 - batch["dones"]) * target_q

        # Compute current Q-values
        q1, q2 = critic.get_both_q_values(batch["observations"], batch["actions"])

        # Compute losses
        q1_loss = jnp.mean((q1 - target_values)*weights ** 2) 
        q2_loss = jnp.mean((q2 - target_values)*weights ** 2) 
        total_loss = q1_loss + q2_loss

        # Info for logging
        info = {
            'q1_loss': q1_loss,
            'q2_loss': q2_loss,
            'q1_mean': jnp.mean(q1),
            'q2_mean': jnp.mean(q2),
            'target_q_mean': jnp.mean(target_q),
        }

        return total_loss, info

    # Compute gradients and update
    (loss, info), grads = nnx.value_and_grad(critic_loss_fn, has_aux=True)(critic)
    info.update({"critic_grad":grads})
    critic_optimizer.update(grads)

    return loss, info

def update_actor(
    actor: Actor,
    critic: DoubleCritic,
    batch: Dict[str, jnp.ndarray],
    actor_optimizer: nnx.Optimizer
) -> Tuple[jnp.ndarray, Dict[str, float]]:
    def actor_loss_fn(actor):
        actions = actor(batch["observations"])
        min_q = critic(batch["observations"], actions)
        actor_loss = -jnp.mean(min_q)
        return actor_loss, {"actor_loss": actor_loss}
    
    (loss, info), grads = nnx.value_and_grad(actor_loss_fn, has_aux=True)(actor)

    actor_optimizer.update(grads)

    return loss, info


def update_target_networks(
    target_net: nnx.Module,
    online_net: nnx.Module,
    tau: float
) -> None:
    """Soft update target networks."""
    target_params = nnx.state(target_net)
    online_params = nnx.state(online_net)

    new_params = jax.tree.map(
        lambda online, target: tau * online + (1 - tau) * target,
        online_params, target_params
    )

    nnx.update(target_net, new_params)



@nnx.jit(static_argnames=('update_policy'))
def update_td3(
    actor: Actor,
    critic: DoubleCritic,
    target_actor: Actor,
    target_critic: DoubleCritic,
    batch: Dict[str, jnp.ndarray],
    gamma: float,
    tau: float,
    policy_noise: float,
    noise_clip: float,
    actor_optimizer: nnx.Optimizer,
    critic_optimizer: nnx.Optimizer,
    key: jax.random.PRNGKey,
    update_policy: bool
) -> Dict[str, float]:
    """
    TD3 update with target policy smoothing.
    """
    info = {}

    critic_loss, critic_info = update_critic(
        critic, target_critic, target_actor, batch,
        gamma, policy_noise, noise_clip, critic_optimizer, key
    )
    info.update({"critic_loss": critic_loss, **critic_info})

    if update_policy:
        actor_loss, actor_info = update_actor(actor, critic, batch, actor_optimizer)
        info.update({"actor_loss": actor_loss, **actor_info})


        update_target_networks(target_critic, critic, tau)
        update_target_networks(target_actor, actor, tau)

    return info


@nnx.jit(static_argnames=('update_policy'))
def update_td3_with_weights(
    actor: Actor,
    critic: DoubleCritic,
    target_actor: Actor,
    target_critic: DoubleCritic,
    weights: jnp.ndarray,
    batch: Dict[str, jnp.ndarray],
    gamma: float,
    tau: float,
    policy_noise: float,
    noise_clip: float,
    actor_optimizer: nnx.Optimizer,
    critic_optimizer: nnx.Optimizer,
    key: jax.random.PRNGKey,
    update_policy: bool
) -> Dict[str, float]:
    """
    TD3 update with target policy smoothing.
    """
    info = {}

    critic_loss, critic_info = update_critic_with_weights(
        critic, target_critic, target_actor, batch,
        gamma, policy_noise, noise_clip, critic_optimizer, weights, key
    )
    info.update({"critic_loss": critic_loss, **critic_info})

    if update_policy:
        actor_loss, actor_info = update_actor(actor, critic, batch, actor_optimizer)
        info.update({"actor_loss": actor_loss, **actor_info})


        update_target_networks(target_critic, critic, tau)
        update_target_networks(target_actor, actor, tau)

    return info