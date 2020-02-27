import numpy as np
from gym import spaces
from RandomTargetEnv import RandomTargetEnv
from stable_baselines.common.vec_env import VecEnv


class RandomTargetVecEnv(VecEnv):
    """
    A vector of environments RandomTargetEnv()
    """
    def __init__(self, num_envs):
        self.num_envs = num_envs 
        self._environments = []
        for i in range(self.num_envs):
            self._environments.append(RandomTargetEnv())

        self.num_obs = 4
        self._observation_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf,
                                             dtype=np.float32)
        #self.num_acts = 4
        #self._action_space = spaces.Box(np.ones(self.num_acts) * -1., np.ones(self.num_acts) * 1., dtype=np.float32)
        self._action_space = spaces.Discrete(4)
        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros((self.num_envs), dtype=np.bool)
        self.rewards = [[] for _ in range(self.num_envs)]

    def step(self, action, visualize=False):
        info = [{} for i in range(self.num_envs)]
        for i in range(self.num_envs):
            self._reward[i] = self._environments[i].step(action[i])
            self._environments[i].observe(self._observation[i])
            self._done[i] = self._environments[i].isTerminalState()
            #self._extraInfo[i] = xxx

            self.rewards[i].append(self._reward[i])
            if self._done[i]:
                eprew = sum(self.rewards[i])
                eplen = len(self.rewards[i])
                epinfo = {"r": eprew, "l": eplen}
                info[i]['episode'] = epinfo
                self.rewards[i].clear()

        return self._observation.copy(), self._reward.copy(), self._done.copy(), info.copy()

    def reset(self):
        for i in range(self.num_envs):
            self._environments[i].reset()
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        return self._observation.copy()

    def reset_and_update_info(self):
        return self.reset(), self._update_epi_info()

    def _update_epi_info(self):
        info = [{} for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            eprew = sum(self.rewards[i])
            eplen = len(self.rewards[i])
            epinfo = {"r": eprew, "l": eplen}
            info[i]['episode'] = epinfo
            self.rewards[i].clear()
        return info

    def render(self, mode='human'):
        raise RuntimeError('This method is not implemented')

    def close():
        pass
    def env_method():
        pass
    def get_attr():
        pass
    def set_attr():
        pass
    def step_async():
        pass
    def step_wait():
        pass

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space
