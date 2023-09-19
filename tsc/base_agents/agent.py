import torch


class Agent(object):
    def __init__(self):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        self.device = torch.device(device)

    def act(self, observation, unava_phase_index):
        raise NotImplementedError

    def observe(self, observation, reward, done, info):
        raise NotImplementedError


class IndependentAgent(Agent):
    def __init__(self, config, obs_act, map_name, thread_number):
        super().__init__()
        self.config = config
        self.agents = dict()

    def act(self, observation):
        acts = dict()
        for agent_id in observation.keys():
            acts[agent_id] = self.agents[agent_id].act(observation[agent_id])
        return acts

    def observe(self, observation, reward, done, info):
        for agent_id in observation.keys():
            self.agents[agent_id].observe(observation[agent_id], reward[agent_id], done, info)
            if done:
                if info['eps'] % 100 == 0:
                    self.agents[agent_id].save(self.config['log_dir']+'agent_'+agent_id)


class SharedAgent(Agent):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.agent = None
        self.valid_acts = None
        self.reverse_valid = None

    def act(self, observation, unava_phase_index):
        batch_obs = observation['pressure']

        batch_acts = self.agent.act(batch_obs, unava_phase_index)
        return batch_acts

    def observe(self, observation, reward, done, info):
        batch_obs = [observation[agent_id] for agent_id in observation.keys()]
        batch_rew = [reward[agent_id] for agent_id in observation.keys()]
        batch_done = [done]*len(batch_obs)
        batch_reset = [False]*len(batch_obs)
        self.agent.observe(batch_obs, batch_rew, batch_done, batch_reset)
        if done:
            if info['eps'] % 100 == 0:
                self.agent.save(self.config['log_dir']+'agent')
