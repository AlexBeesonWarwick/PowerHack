from grid2op.gym_compat import GymEnv
import grid2op
from gym import Env
from gym.utils.env_checker import check_env

try:
    from lightsim2grid import LightSimBackend

    bk_cls = LightSimBackend
except ImportError as exc:
    print(f"Error: {exc} when importing faster LightSimBackend")
    from grid2op.Backend import PandaPowerBackend

    bk_cls = PandaPowerBackend

env_name = "rte_case5_example"

### Training environment ###
training_env = grid2op.make(env_name, test=True, backend=bk_cls())
env = GymEnv(training_env)

isinstance(env, Env)
check_env(env, warn=False)

from grid2op.gym_compat import DiscreteActSpace
env.action_space = DiscreteActSpace(training_env.action_space,
                                        attr_to_keep=["set_bus"])

from grid2op.gym_compat import BoxGymObsSpace
env.observation_space = BoxGymObsSpace(training_env.observation_space,
                                           attr_to_keep=["rho"])

### Testing environment ###
testing_env = grid2op.make(env_name, test=True, backend=bk_cls())  # we put "test=True" in this notebook because...
# it's a notebook to explain things. Of course, do not put "test=True" if you really want
# to train an agent...
gym_env_test = GymEnv(testing_env)

isinstance(gym_env_test, Env)
check_env(gym_env_test, warn=False)

from grid2op.gym_compat import DiscreteActSpace
gym_env_test.action_space = DiscreteActSpace(testing_env.action_space,
                                        attr_to_keep=["set_bus"])

from grid2op.gym_compat import BoxGymObsSpace
gym_env_test.observation_space = BoxGymObsSpace(testing_env.observation_space,
                                           attr_to_keep=["rho"])

# Import modules
import sys
import gym
import torch
import numpy as np
import random

sys.path.append('YourPathHere')
import SAC_Discrete

# Hyperparameters
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
device = "cuda:0"
batch_size = 256

num_episodes = 1000
replay_buffer = []
replay_buffer_size = 1e6
env_steps = 0
eval_every = 1000

agent = SAC_Discrete.Agent(state_dim, action_dim, device=device)

for episode in range(num_episodes):
    # Training #
    done = False
    state = env.reset()
    score_train = 0
    while not done:
        action = agent.choose_action(state, sample=True)
        next_state, reward, done, info = env.step(action.item())
        score_train += reward
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        env_steps += 1

        if len(replay_buffer) > batch_size:
            agent.train(replay_buffer)

        if env_steps % eval_every == 0:
            # Evaluation #
            done_eval = False
            state_eval = gym_env_test.reset()
            score_eval = 0
            length_episode = 0
            while not done_eval:
                with torch.no_grad():
                    action_eval = agent.choose_action(state_eval)
                    state_eval, reward_eval, done_eval, info_eval = gym_env_test.step(action_eval.item())
                    score_eval += reward_eval
                    length_episode += 1

            print("Episode", episode, "Env interactions", env_steps,
                  "Score Eval %.2f" % score_eval, "Eps length", length_episode)
