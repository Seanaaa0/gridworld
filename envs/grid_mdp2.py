import logging
import random
import numpy as np
from gym import spaces
import gym

logger = logging.getLogger(__name__)


class GridEnv4x4(gym.Env):
    def __init__(self, render_mode=None):
        self.render_mode = render_mode

        self.rows = 4
        self.cols = 4
        self.n_states = self.rows * self.cols
        self.states = list(range(1, self.n_states + 1))

        self.actions = ['n', 'e', 's', 'w']
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Discrete(self.n_states)

        self.cell_size = 80
        self.screen_width = 600
        self.screen_height = 400
        self.start_x = (self.screen_width - self.cols * self.cell_size) // 2
        self.start_y = (self.screen_height - self.rows * self.cell_size) // 2
        self.x = [(self.start_x + (i % self.cols) * self.cell_size +
                   self.cell_size // 2) for i in range(self.n_states)]
        self.y = [(self.start_y + (self.rows - 1 - (i // self.cols)) *
                   self.cell_size + self.cell_size // 2) for i in range(self.n_states)]

        self.rewards = {}
        self.t = {}
        self.terminate_states = {}
        self.goal_state = None
        self.max_attempts = 25
        self.attempts = 0

        self._generate_transitions()
        self._randomize_terminal()
        self._set_default_rewards()

        self.gamma = 0.8
        self.viewer = None
        self.state = None

    def _set_default_rewards(self):
        step_penalty = -0.01  # 小小的懲罰來避免 agent 一直閒晃
        for s in self.states:
            if s in self.terminate_states:
                continue
            for a in self.actions:
                key = f'{s}_{a}'
                if key not in self.rewards:
                    self.rewards[key] = step_penalty

    def _generate_transitions(self):
        for s in self.states:
            r, c = (s - 1) // self.cols, (s - 1) % self.cols
            if r > 0:
                self.t[f'{s}_n'] = s - self.cols
            if r < self.rows - 1:
                self.t[f'{s}_s'] = s + self.cols
            if c > 0:
                self.t[f'{s}_w'] = s - 1
            if c < self.cols - 1:
                self.t[f'{s}_e'] = s + 1

    def _randomize_terminal(self):
        candidates = self.states.copy()
        self.goal_state = random.choice(candidates)
        self.terminate_states[self.goal_state] = 'goal'
        self.rewards.update(
            {f'{self.goal_state}_{a}': 1.0 for a in self.actions})
        candidates.remove(self.goal_state)

        pits = random.sample(candidates, k=min(2, len(candidates)))
        for pit in pits:
            self.terminate_states[pit] = 'pit'
            self.rewards.update({f'{pit}_{a}': -1.0 for a in self.actions})

    def getStates(self):
        return self.states

    def getAction(self):
        return self.actions

    def getTerminate_states(self):
        return self.terminate_states

    def setAction(self, s):
        self.state = s

    def step(self, action_index):
        action_str = self.actions[action_index]
        return self._step(action_str)

    def _step(self, action):
        state = self.state
        key = f"{state}_{action}"
        next_state = self.t.get(key, state)
        reward = self.rewards.get(key, 0.0)

        if self.terminate_states.get(next_state) == 'goal':
            self.state = next_state
            return next_state, reward, True, {}

        if self.terminate_states.get(next_state) == 'pit':
            self.attempts += 1
            if self.attempts >= self.max_attempts:
                return next_state, reward, True, {}
            valid = [
                s for s in self.states if s not in self.terminate_states or self.terminate_states[s] != 'goal']
            self.state = random.choice(valid)
            return self.state, reward, False, {}

        self.state = next_state
        return next_state, reward, False, {}

    def reset(self):
        self.attempts = 0
        valid = [
            s for s in self.states if s not in self.terminate_states or self.terminate_states[s] != 'goal']
        self.state = random.choice(valid)
        return self.state

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(
                self.screen_width, self.screen_height)

            for i in range(self.rows + 1):
                y = self.start_y + i * self.cell_size
                self.viewer.add_geom(rendering.Line(
                    (self.start_x, y), (self.start_x + self.cols * self.cell_size, y)))

            for j in range(self.cols + 1):
                x = self.start_x + j * self.cell_size
                self.viewer.add_geom(rendering.Line(
                    (x, self.start_y), (x, self.start_y + self.rows * self.cell_size)))

            self.terminal_geoms = {}
            for s in self.terminate_states:
                circle = rendering.make_circle(30)
                trans = rendering.Transform(
                    translation=(self.x[s - 1], self.y[s - 1]))
                circle.add_attr(trans)
                if self.terminate_states[s] == 'goal':
                    circle.set_color(1, 0.9, 0)  # gold
                else:
                    circle.set_color(0, 0, 0)  # pit
                self.viewer.add_geom(circle)
                self.terminal_geoms[s] = circle

            self.agent = rendering.make_circle(25)
            self.agent_trans = rendering.Transform()
            self.agent.add_attr(self.agent_trans)
            self.agent.set_color(0.4, 0.6, 0.8)
            self.viewer.add_geom(self.agent)

        if self.state is not None:
            self.agent_trans.set_translation(
                self.x[self.state - 1], self.y[self.state - 1])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
