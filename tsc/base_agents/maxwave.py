import numpy as np
from tsc.base_agents.agent import SharedAgent, Agent


class MAXWAVE(SharedAgent):
    def __init__(self, config):
        super().__init__(config)
        self.agent = WaveAgent(signal_configs[map_name]['phase_pairs'])


class WaveAgent(Agent):
    def __init__(self, phase_pairs):
        super().__init__()
        self.phase_pairs = phase_pairs

    def act(self, observations, unava_phase_index):
        acts = []
        for i, observation in enumerate(observations):
            all_press = []
            for pair in self.phase_pairs:
                all_press.append(observation[pair[0]] + observation[pair[1]])
            all_press = np.array(all_press)
            all_press[unava_phase_index[i]] = -1e8
            acts.append(np.argmax(all_press))
        return acts

    def observe(self, observation, reward, done, info):
        pass

    def save(self, path):
        pass