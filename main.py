from env.create_env import create_env_base, DecMAPFConfig
from mcts_cpp.cppmcts import MCTSConfig, MCTSInference, mcts_preprocessor


def example():
    env_cfg = DecMAPFConfig(with_animation=True, num_agents=8, seed=0, map_name='pico_s21_od30_na32',
                            max_episode_steps=64)

    algo = MCTSInference(MCTSConfig(num_expansions=250, gamma=0.96))
    env = mcts_preprocessor(create_env_base(env_cfg))

    obs, _ = env.reset(seed=env.grid_config.seed)
    while True:
        obs, rew, dones, tr, infos = env.step(algo.act(obs))
        if all(dones) or all(tr):
            break

    print(infos[0]['metrics'])


if __name__ == '__main__':
    example()
