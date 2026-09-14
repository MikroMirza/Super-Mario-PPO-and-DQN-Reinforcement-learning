import os

import gym
import numpy as np
import vizdoom as vzd
from gym import spaces


class VizDoomEnv(gym.Env):
    """Tanak gym.Env omotac oko sirovog ViZDoom API-ja (stari gym 4-tuple API)."""

    _MONSTER_NAMES = {"Zombieman", "ShotgunGuy", "ChaingunGuy", "MarineChainsawVzd",
                       "MarineBFG", "MarineChainsaw", "MarinePistol", "MarineRocket",
                       "MarineSSG", "MarineChaingun", "MarinePlasma", "MarineBerserk",
                       "MarineRailgun"}

    def _enemy_visible(self, state):
        if state is None or not state.labels:
            return False
        return any(l.object_name in self._MONSTER_NAMES for l in state.labels)

    def __init__(self, config_path, frame_repeat=1, window_visible=False,
                 reward_mode="kills", kill_reward=1.0, distance_scale=0.03,
                 health_scale=0.05, aim_reward=0.5, aim_penalty=0.1,
                 longevity_reward=0.01, extra_combos=None, stuck_penalty=0.0,
                 allowed_button_indices=None, distance_discount=1.0,
                 gate_advance_on_enemy=False, gated_discount=0.05, exploration_scale=0.01
                #    gate_distance=250.0
                   ):
        super().__init__()

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"ViZDoom config nije pronadjen: {config_path}")
    
        self.frame_repeat = frame_repeat
        self.reward_mode = reward_mode
        self.kill_reward = kill_reward
        self.distance_scale = distance_scale
        self.health_scale = health_scale
        self.aim_reward = aim_reward
        self.aim_penalty = aim_penalty
        self.longevity_reward = longevity_reward
        self.stuck_penalty = stuck_penalty
        self.distance_discount = distance_discount
        # gate_advance_on_enemy: kad je True, SVA nagrada za kretanje (i
        # ugradjena WAD nagrada i nasa distance_scale) se dodatno prigusuje
        # na 'gated_discount' (npr. 5%) SVAKI PUT kad je zivi neprijatelj
        # vidljiv u kadru. Ovo direktno vezuje "napredak koji se isplati" za
        # "nema vise zivih neprijatelja koje vidis" - prisiljava agenta da
        # prvo ocisti pre nego sto trcanje uopste ima smisla.
        self.gate_advance_on_enemy = gate_advance_on_enemy
        self.gated_discount = gated_discount
        # self.gate_distance = gate_distance
        
        self.game = vzd.DoomGame()
        self.game.load_config(config_path)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_window_visible(window_visible)
        self.game.set_labels_buffer_enabled(True)
        self.game.init()

        n_buttons = self.game.get_available_buttons_size()
        self.exploration_scale = exploration_scale

        self._visited_cells = set()
        self._exploration_cell_size = 64

        # allowed_button_indices: ogranicava koji pojedinacni dugmici se
        # koriste za osnovne (one-hot) akcije. Ostala dugmad ostaju
        # deklarisana u .cfg (ViZDoom to zahteva/ne smeta), samo se nikad ne
        # pritiskaju sama. Npr. za my_way_home zvanicna specifikacija scenarija
        # koristi samo TURN_LEFT/TURN_RIGHT/MOVE_FORWARD (indeksi 0,1,2), iako
        # nas .cfg fajl (iz pip paketa) deklarise i MOVE_LEFT/MOVE_RIGHT (3,4).
        # Bez ovog filtera, agent ima 2 dodatne "besplatne" akcije (strafe bez
        # ikakvog rizika) koje se lako pretvore u beskorisno levo-desno ljuljanje.
        base_indices = allowed_button_indices if allowed_button_indices is not None else list(range(n_buttons))
        self._actions = []
        for idx in base_indices:
            vec = [0] * n_buttons
            vec[idx] = 1
            self._actions.append(vec)

        if extra_combos:
            for combo in extra_combos:
                vec = [0] * n_buttons
                for idx in combo:
                    vec[idx] = 1
                self._actions.append(vec)

        self.action_space = spaces.Discrete(len(self._actions))

        buttons = self.game.get_available_buttons()
        self._attack_idx = buttons.index(vzd.Button.ATTACK) if vzd.Button.ATTACK in buttons else None
        self._forward_idx = buttons.index(vzd.Button.MOVE_FORWARD) if vzd.Button.MOVE_FORWARD in buttons else None

        h = self.game.get_screen_height()
        w = self.game.get_screen_width()
        self.observation_space = spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)

        self._prev_kills = 0
        self._prev_x = 0.0
        self._max_dist_from_start = 0.0
        self._start_x = 0.0
        self._start_y = 0.0
        self._tick_prev_x = 0.0
        self._tick_prev_y = 0.0
        self._prev_health = 100.0
        self._prev_hits = 0.0
        self._healthpacks_picked = 0
        self._health_gained = 0.0

    def _get_state_and_obs(self):
        state = self.game.get_state()
        if state is None:
            return None, np.zeros(self.observation_space.shape, dtype=np.uint8)
        return state, state.screen_buffer

    def reset(self, **kwargs):
        self.game.new_episode()
        self._prev_kills = 0
        self._prev_x = self.game.get_game_variable(vzd.GameVariable.POSITION_X)
        self._start_x = self._prev_x
        self._start_y = self.game.get_game_variable(vzd.GameVariable.POSITION_Y)
        self._tick_prev_x = self._start_x
        self._tick_prev_y = self._start_y
        self._max_dist_from_start = 0.0
        self._prev_health = self.game.get_game_variable(vzd.GameVariable.HEALTH)
        self._prev_hits = self.game.get_game_variable(vzd.GameVariable.HITCOUNT)
        self._healthpacks_picked = 0
        self._health_gained = 0.0
        self._visited_cells = set()
        
        _, obs = self._get_state_and_obs()
        return obs

    def step(self, action):
        action_vec = self._actions[action]
        base_reward = self.game.make_action(action_vec, self.frame_repeat)

        done = self.game.is_episode_finished()
        state, obs = self._get_state_and_obs()

        health = self.game.get_game_variable(vzd.GameVariable.HEALTH)
        died = bool(done and health <= 0)
        attacked = bool(self._attack_idx is not None and action_vec[self._attack_idx] == 1)


        if not done:
            effective_discount = self.distance_discount
            if self.gate_advance_on_enemy and self._enemy_visible(state):
                effective_discount = self.gated_discount
            reward = base_reward * effective_discount
        else:
            effective_discount = self.distance_discount
            reward = base_reward

        health_delta = health - self._prev_health


        if health_delta > 0:
            self._healthpacks_picked += 1
            self._health_gained += health_delta

        reward += self.health_scale * health_delta
        self._prev_health = health


        kills = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        kill_delta = kills - self._prev_kills
        self._prev_kills = kills
        reward += self.kill_reward * kill_delta

        if not died:
            reward += self.longevity_reward
        
        # STUCK PENALTY: ako je agent pokusao da ide napred (MOVE_FORWARD u
        # akciji) ali se pozicija ovog tika skoro uopste nije promenila,
        # verovatno je udario u zid. Kaznjava se SAMO pokusaj kretanja -
        # cisto okretanje u mestu (bez FORWARD komponente) se ne kaznjava,
        # jer to nije "zaglavljivanje", moze biti namerno razgledanje.
        if not done and self._forward_idx is not None and action_vec[self._forward_idx] == 1:
            x = self.game.get_game_variable(vzd.GameVariable.POSITION_X)
            y = self.game.get_game_variable(vzd.GameVariable.POSITION_Y)
            moved = ((x - self._tick_prev_x) ** 2 + (y - self._tick_prev_y) ** 2) ** 0.5
            if moved < 0.5:
                reward -= self.stuck_penalty
            self._tick_prev_x, self._tick_prev_y = x, y
        elif not done:
            self._tick_prev_x = self.game.get_game_variable(vzd.GameVariable.POSITION_X)
            self._tick_prev_y = self.game.get_game_variable(vzd.GameVariable.POSITION_Y)

        aimed = False
        if attacked and not done:
            hits = self.game.get_game_variable(vzd.GameVariable.HITCOUNT)
            hit_landed = hits > self._prev_hits
            self._prev_hits = hits
            aimed = hit_landed
            reward += self.aim_reward if hit_landed else -self.aim_penalty
        else:
            self._prev_hits = self.game.get_game_variable(vzd.GameVariable.HITCOUNT)

        x = self.game.get_game_variable(vzd.GameVariable.POSITION_X)
        y = self.game.get_game_variable(vzd.GameVariable.POSITION_Y)

        cell_x = int(x // self._exploration_cell_size)
        cell_y = int(y // self._exploration_cell_size)

        cell = (cell_x, cell_y)

        if cell not in self._visited_cells:
            self._visited_cells.add(cell)
            reward += self.exploration_scale

        info = {
            "died": died,
            "kills": int(self._prev_kills),
            "attacked": attacked,
            "aimed": aimed,

            "healthpacks_picked": self._healthpacks_picked,
            "health_gained": float(self._health_gained),
            "health": float(health),
        }

        if self.reward_mode == "distance":
            x = self.game.get_game_variable(vzd.GameVariable.POSITION_X)
            gate = effective_discount if not done else 1.0
            reward += self.distance_scale * gate * (x - self._prev_x)
            self._prev_x = x
            info["x_pos"] = float(x)

        elif self.reward_mode == "explore":
            x = self.game.get_game_variable(vzd.GameVariable.POSITION_X)
            y = self.game.get_game_variable(vzd.GameVariable.POSITION_Y)

            dist = ((x - self._start_x) ** 2 + (y - self._start_y) ** 2) ** 0.5

            new_record = max(0.0, dist - self._max_dist_from_start)

            reward += self.exploration_scale * new_record

            self._max_dist_from_start = max(self._max_dist_from_start,dist)
            info["x_pos"] = float(self._max_dist_from_start)

        return obs, reward, done, info

    def render(self):
        pass

    def close(self):
        self.game.close()