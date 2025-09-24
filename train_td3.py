"""
TD3 training example following CleanRL's style.
"""
import os
import random
import numpy as np
import os
import jax
from flax import nnx
import jax.numpy as jnp
from typing import Dict
import gymnasium as gym
import dataclasses
import tyro
import time
from cleanrl_jax.agents.td3.td3_leaner import TD3Learner
from cleanrl_jax.utils.ReplayBuffer import ReplayBuffer
from cleanrl_jax.utils.dmc_wrapper import make_env_dmc
import cleanrl_jax.utils.logger as logger
from cleanrl_jax.agents.td3.network import DoubleCritic, Actor
from cleanrl_jax.utils.Normalization import ObservationNormalizerWrapper
from cleanrl_jax.utils.grad_monitor import GradientMonitor
from cleanrl_jax.utils.PrioritizedReplayBuffer import PrioritizedReplayBuffer


os.environ['MUJOCO_GL'] = 'egl'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false' 

@dataclasses.dataclass
class Config:
    env_id: str = "dmc-humanoid-run"
    seed: int = 1
    num_envs: int = 1
    total_timesteps: int = int(1e6)
    buffer_size: int = int(1e6)
    policy_frequency: int = 2
    learning_starts: int = int(25e3)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    lr: float = 3e-4
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    exploration_noise: float = 0.1

    normalize_observation: bool = True

    linear_decay_steps: int = 80000 # 0 means uniform sampling

    replay_ratio: int = 1  
    # Update frequency for every environment step

    # SimBA network architecture parameters
    actor_hidden_dim: int = 128
    actor_num_blocks: int = 1
    actor_block_type: str = "residual"
    critic_hidden_dim: int = 512
    critic_num_blocks: int = 2
    critic_block_type: str = "residual"

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
def compute_td_errors(critic: DoubleCritic, target_critic: DoubleCritic, actor: Actor, batch: Dict[str, jnp.ndarray], gamma: float):
    next_actions = actor(batch["next_observations"])
    min_next_q = target_critic(batch["next_observations"], next_actions)
    target_q = batch["rewards"] + (1 - batch["dones"]) * gamma * min_next_q
    q1 = critic.critic1(batch["observations"], batch["actions"])
    q2 = critic.critic2(batch["observations"], batch["actions"])
    td_errors = (jnp.abs(q1 - target_q) + jnp.abs(q2 - target_q)) / 2
    return td_errors

def main():

    print("🚀 TD3 training")
    print("=" * 60)

    args = tyro.cli(Config)
    name = f"linear_decay_{args.linear_decay_steps}" if not args.prioritized_replay else f"prioritized_replay"
    logger.init(project=f"{args.env_id}", name=name, config=vars(args),dir=f"Results/td3/utd_{args.replay_ratio}")
    np.random.seed(args.seed)
    random.seed(args.seed)



    env = [make_env(args.env_id, args.seed + i) for i in range(args.num_envs)]
    envs = gym.vector.SyncVectorEnv(env)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    print(f"Environment: {args.env_id}")
    print(f"Observation space: {envs.single_observation_space.shape}")
    print(f"Action space: {envs.single_action_space.shape}")
    print(f"Total training steps: {args.total_timesteps:,}")


    td3_learner = TD3Learner(
        env=envs,
        seed=args.seed,
        lr=args.lr,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        gamma=args.gamma,
        tau=args.tau,
        policy_frequency=args.policy_frequency,
        actor_hidden_dim=args.actor_hidden_dim,
        actor_num_blocks=args.actor_num_blocks,
        actor_block_type=args.actor_block_type,
        critic_hidden_dim=args.critic_hidden_dim,
        critic_num_blocks=args.critic_num_blocks,
        critic_block_type=args.critic_block_type
    )

    if args.normalize_observation:
        td3_learner = ObservationNormalizerWrapper(td3_learner,envs.single_observation_space.shape)


    if args.prioritized_replay:
        rb = PrioritizedReplayBuffer.from_env(envs, max_size=args.buffer_size)
    else:
        rb = ReplayBuffer.from_env(envs, max_size=args.buffer_size, linear_decay_steps=args.linear_decay_steps)

    print(f"Buffer size: {args.buffer_size:,}")
    print(f"Start training...")
    print("=" * 60)
    
    grad_monitor = GradientMonitor(monitor_frequency=1000, sparsity_threshold=1e-8) # monitor the gradient

    start_time = time.time()

    obs,_  = envs.reset()

    for global_step in range(args.total_timesteps):
        # action selection
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(args.num_envs)])
        else:
            actions = td3_learner.get_action(obs)

            noise = jax.random.normal(jax.random.PRNGKey(global_step), actions.shape) * args.exploration_noise * td3_learner.actor.action_scale
            actions = actions + noise

            # 确保动作在合理范围内
            actions = jnp.clip(actions, td3_learner.actor.action_low, td3_learner.actor.action_high)

        # environment interaction
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)



        # record episode results - simplified output
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
                        "episode_length": episode_length,
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
                for _ in range(args.replay_ratio):
                    batch, indices, weights = rb.sample(args.batch_size)
                    td_errors = compute_td_errors(td3_learner.critic, td3_learner.target_critic, td3_learner.actor, batch, args.gamma)
                    info = td3_learner.update(batch, weights)
                    rb.update_priorities(indices, np.array(td_errors).flatten())
                    stats = grad_monitor.monitor_gradients(info["critic_grad"])
                    if stats:
                        logger.log(stats,global_step)
            else:
                for _ in range(args.replay_ratio):
                    batch = rb.sample(args.batch_size)
                    info = td3_learner.update(batch)
                    stats = grad_monitor.monitor_gradients(info["critic_grad"])
                    if stats:
                        logger.log(stats,global_step)

    # training completed
    envs.close()
    logger.finish()


if __name__ == "__main__":
    main()