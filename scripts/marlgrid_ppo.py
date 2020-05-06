import torch
import numpy as np
import torch.nn as nn

import datetime, time

from kamarl.ppo import PPOAgent
from kamarl.utils import find_cuda_device

from marlgrid import envs as marl_envs
from marlgrid.agents import IndependentLearners, LearningAgent
from marlgrid.utils.video import GridRecorder


class AgentWrapper(LearningAgent):
    def __init__(self, learner_kwargs, agent_hook):
        super().__init__(**learner_kwargs)
        self._agent = agent_hook(self.observation_space, self.action_space)

    def set_device(self, dev):
        self._agent.set_device(dev)

    def action_step(self, obs):
        return self._agent.action_step(obs)

    def save_step(self, *values, **kwvalues):
        return self._agent.save_step(*values, **kwvalues)

    def start_episode(self):
        return self._agent.start_episode()

    def end_episode(self):
        return self._agent.end_episode()


marlgrid_agent_kwargs = {
    'view_tile_size': 3,
    'view_size': 7,
}

agents = IndependentLearners(
    AgentWrapper(marlgrid_agent_kwargs, agent_hook=PPOAgent),
)

# wbl = WandbLogger(name='ppo', project='ppo_test')
wbl = None


device = find_cuda_device('1080 Ti')
# device = find_cuda_device('1070')


count_parameters = lambda mod: np.sum([np.prod(x.shape) for x in mod.parameters()])
print(count_parameters(agents[0]._agent.ac))
# exit()
for agent in agents:
    agent.set_device(device)

grid_size = 20
clutter_density = 0.2
max_steps = 300
n_clutter = int(clutter_density * (grid_size-2)**2)

grid_params = {
    'grid_size': grid_size,
    'max_steps': max_steps,
    'seed': 1,
    'randomize_goal': True,
    'n_clutter': n_clutter
}
env = marl_envs.ClutteredMultiGrid(agents, **grid_params)

# grid_params = {'grid_size': 20, 'max_steps': 50}
# env = marl_envs.EmptyMultiGrid(agents, **grid_params)


# wbl.log_hyperparams({
#     'env_name': env.__class__.__name__,
#     'env_params': grid_params})

env = GridRecorder(env)  # , render_kwargs = {'show_agent_views': False})

run_time = datetime.datetime.now().strftime("%m_%d_%H:%M:%S")

total_reward = 0
num_episodes = int(1e6)
for ep_num in range(num_episodes):
    # Initialize the environment and state
    obs_array = env.reset()
    done = False
    with agents.episode():
        with torch.set_grad_enabled(False):
            ep_start_time = time.time()

            env.recording = ep_num % 500 == 0
            ep_steps = 0
            # env.render(show_agent_views=True)
            agent_total_rewards = None
            while not done:

                # Get an action for each agent.
                # action_array = [agent.action_space.sample() for agent]
                action_array = agents.action_step(obs_array)

                next_obs_array, reward_array, done_array, _ = env.step(action_array)
                # if agent_total_rewards is None:
                #     agent_total_rewards = 1 * reward_array
                # else:
                #     agent_total_rewards += reward_array

                # any(transitions[2]) is competitive; the episode will end as soon as the first agent reaches the goal
                # all(transitions[2]) is less competitive; all agents have a chance to obtain the goal before the timeout
                done = all(done_array)

                total_reward += reward_array.sum()
                # Penalize agents that don't get a reward during the episode
                # if done:
                #     reward_array = reward_array - 1.0 * (agent_total_rewards == 0)

                agents.save_step(obs_array, action_array, reward_array, done_array)

                obs_array = next_obs_array

                ep_steps += 1
                # env.render(show_agent_views=True)

                # wbl.flush_values(step=True)


            # Save the last step twice
            # agents.save_step(obs_array, action_array, reward_array, done_array)

            ep_time = time.time() - ep_start_time
            if env.recording:
                env.export_video(
                    f"/fast/multigrid3/run_{run_time}/episode_{ep_num}.mp4",
                    render_frame_images=True,
                )

            # if ep_num>0 and ep_num%10==0:
            #     pdb.set_trace()

            print(
                f"Episode {ep_num: >5d}: {ep_steps: <4d} ticks | cum rew={total_reward: <4.1f} | fps={ep_steps/ep_time: >5.2f}"
            )
            # for k, a in enumerate(agents):
            #     print(f" > {k} ", ', '.join(str(x) for x in a.ep_act_hist))
                
