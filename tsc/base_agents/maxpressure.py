from tsc.base_agents.agent import SharedAgent
from tsc.base_agents.maxwave import WaveAgent
from sumo_files.scenarios.resco_envs.signal_config import signal_configs


class MAXPRESSURE(SharedAgent):
    def __init__(self, config, phase_pairs):
        super().__init__(config)
        self.agent = MaxAgent(phase_pairs)


class MaxAgent(WaveAgent):
    def act(self, observation, unava_phase_index):
        return super().act(observation, unava_phase_index)





