"""
SAC training following CleanRL's style.
"""

import random
import numpy as np
import gymnasium as gym
import dataclasses
import tyro
import time
from cleanrl_jax.agents.sac.sac_leaner import SACLearner
from cleanrl_jax.utils.ReplayBuffer import ReplayBuffer
from cleanrl_jax.utils.dmc_wrapper import make_env_dmc
import cleanrl_jax.utils.logger as logger
from cleanrl_jax.utils.Normalization import ObservationNormalizerWrapper
from cleanrl_jax.utils.grad_monitor import GradientMonitor
from cleanrl_jax.agents.sac.network import DoubleCritic, Actor
import jax
import jax.numpy as jnp
import os
from cleanrl_jax.utils.PrioritizedReplayBuffer import PrioritizedReplayBuffer
from typing import Dict
from flax import nnx

# environment variables
os.environ['MUJOCO_GL'] = 'egl'                        # use egl for mujoco
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # disable preallocation




@dataclasses.dataclass
class Config:
    env_id: str = "dmc-humanoid-walk"
    seed: int = 1
    num_envs: int = 1
    total_timesteps: int = int(1e6)
    buffer_size: int = int(1e6)
    policy_frequency: int = 2
    target_network_frequency: int = 1
    learning_starts: int = 5000
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    policy_lr: float = 1e-4
    q_lr: float = 1e-4
    alpha: float = 0.2  
    autotune: bool = False

    # Linear decay parameters
    linear_decay_steps: int = 80000  # 0 means no linear decay (Uniform sampling)

    # SAC replay parameters
    replay_ratio: int = 1 # how many times to replay the batch

    # SimBA network architecture parameters
    actor_hidden_dim: int = 128
    actor_num_blocks: int = 1
    actor_block_type: str = "residual"
    critic_hidden_dim: int = 512
    critic_num_blocks: int = 2
    critic_block_type: str = "residual"

    # Normalization parameters
    normalize_observation: bool = True

    # Prioritized replay parameters
    prioritized_replay: bool = False



def make_env(env_id: str, seed: int):
    """Create environment."""
    def thunk():
        if env_id[:3] == "dmc":
            env = make_env_dmc(env_id[4:])
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        if hasattr(env, 'seed'):
            env.seed(seed)
        return env
    return thunk



@nnx.jit()
def compute_td_errors(critic:DoubleCritic, target_critic:DoubleCritic, actor: Actor, batch:Dict[str,jnp.ndarray], alpha_value:float, gamma:float, key:jax.random.PRNGKey):
    """Compute TD errors for a batch of experiences."""
    next_actions, next_log_pi = actor.get_action(batch["next_observations"], key)
    min_next_q = target_critic(batch["next_observations"],next_actions)

    target_q = batch["rewards"] + (1 - batch["dones"]) * gamma * (min_next_q - alpha_value * next_log_pi)
    q1 = critic.critic1(batch["observations"], batch["actions"])
    q2 = critic.critic2(batch["observations"], batch["actions"])
    td_errors = (jnp.abs(q1 - target_q) + jnp.abs(q2 - target_q)) / 2
    return td_errors


def main():
    
    print("🚀 SAC training")
    print("=" * 60)

    args = tyro.cli(Config)
    name = f"linear_decay_{args.linear_decay_steps}" if not args.prioritized_replay else f"prioritized_replay"
    logger.init(project=f"{args.env_id}", name=name, config=vars(args),dir=f"Results/sac/utd_{args.replay_ratio}")

    np.random.seed(args.seed)
    random.seed(args.seed)



    # environment setting
    env = [make_env(args.env_id, args.seed + i) for i in range(args.num_envs)]
    envs = gym.vector.SyncVectorEnv(env)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    print(f"Environment: {args.env_id}")
    print(f"Observation space: {envs.single_observation_space.shape}")
    print(f"Action space: {envs.single_action_space.shape}")
    print(f"Total training steps: {args.total_timesteps:,}")

    # SAC learner
    sac_learner = SACLearner(
        env=envs,
        seed=args.seed,
        policy_lr=args.policy_lr,
        q_lr=args.q_lr,
        gamma=args.gamma,
        tau=args.tau,
        policy_frequency=args.policy_frequency,
        target_network_frequency=args.target_network_frequency,
        alpha=args.alpha,
        autotune=args.autotune,
        actor_hidden_dim=args.actor_hidden_dim,
        actor_num_blocks=args.actor_num_blocks,
        actor_block_type=args.actor_block_type,
        critic_hidden_dim=args.critic_hidden_dim,
        critic_num_blocks=args.critic_num_blocks,
        critic_block_type=args.critic_block_type
    )
    grad_monitor = GradientMonitor(monitor_frequency=1000, sparsity_threshold=1e-8)
    if args.normalize_observation:
        sac_learner = ObservationNormalizerWrapper(sac_learner, envs.single_observation_space.shape)  # normalize the observation

    if args.prioritized_replay:
        rb = PrioritizedReplayBuffer.from_env(envs, max_size=args.buffer_size)
    else:
        rb = ReplayBuffer.from_env(envs, max_size=args.buffer_size, linear_decay_steps=args.linear_decay_steps)

    print(f"Buffer size: {args.buffer_size:,}")
    print(f"Start training...")
    print("=" * 60)

    start_time = time.time()

    obs,_  = envs.reset(seed=args.seed)

    for global_step in range(args.total_timesteps):
        # action selection
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(args.num_envs)])
        else:
            actions = sac_learner.get_action(obs)

        # environment interaction
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    episode_return = info['episode']['r'].item()
                    episode_length = info['episode']['l'].item()

                    current_time = time.time()
                    total_time = current_time - start_time
                    avg_speed = (global_step + 1) / total_time if total_time > 0 else 0

                    print(f"🎉 Step {global_step:,}: Episode return {episode_return:.1f} (length: {episode_length}) "
                          f"⚡ {avg_speed:.1f} steps/s")

                    logger.log({
                        "episode_return": episode_return,
                        "episode_length": episode_length
                    },global_step,commit=False)



        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        # add to replay buffer
        rb.add(obs, actions, rewards, real_next_obs, terminations)

        obs = next_obs
        # training update
        if global_step >= args.learning_starts:
            if args.prioritized_replay:
                for _ in range(args.replay_ratio): # replay the batch
                    batch, indices, weights = rb.sample(args.batch_size)
                    info = sac_learner.update(batch,weights=weights)
                    td_errors = compute_td_errors(sac_learner.qf, sac_learner.qf_target, sac_learner.actor, batch, sac_learner.alpha(), sac_learner.gamma, jax.random.PRNGKey(global_step))
                    rb.update_priorities(indices, np.array(td_errors).flatten())
                    stats = grad_monitor.monitor_gradients(info["critic_grad"])
                    if stats:
                        logger.log(stats,global_step)
            else:
                batch = rb.sample(args.batch_size)
                for _ in range(args.replay_ratio): # replay the batch
                    info = sac_learner.update(batch)
                    stats = grad_monitor.monitor_gradients(info["critic_grad"])
                    if stats:
                        logger.log(stats,global_step)


    # training completed
    envs.close()
    logger.finish()



if __name__ == "__main__":
    main()
