from . import from_gym


class FromRlp(from_gym.FromGym):

    def __init__(self, env, obs_key="image", act_key="action", **kwargs):
        super().__init__(env, obs_key, act_key)
        self.current_step = 0
        self._max_episode_steps = kwargs.get("max_episode_steps", 10000)

    def step(self, action):
        if action["reset"] or self._done:
            self._done = False
            self.current_step = 0
            obs, _ = self._env.reset()
            return self._obs(obs, 0.0, is_first=True)
        if self._act_dict:
            action = self._unflatten(action)
        else:
            action = action[self._act_key]
        obs, reward, self._done, truncated, self._info = self._env.step(action)
        self.current_step += 1
        self._done = self._done or self.current_step >= self._max_episode_steps
        self._done = self._done or truncated
        return self._obs(
            obs,
            reward,
            is_last=bool(self._done),
            is_terminal=bool(self._info.get("is_terminal", self._done)),
        )
