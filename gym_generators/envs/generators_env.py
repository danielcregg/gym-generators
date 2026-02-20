"""
Combined Economic and Emission Dispatch (CEED) Gymnasium Environment.

This module implements a custom Gymnasium environment for training reinforcement
learning agents to solve the 10-generator, 24-hour combined economic and emission
dispatch problem. The agent controls 9 generators (units 2-10) while unit 1 serves
as the slack generator that balances supply and demand.

Bugs fixed from original:
    1. Unreachable termination condition (n==11 check after n reset)
    2. Inconsistent states_array indexing
    3. Math bug in get_p_1_m (P_n*B[n][j]*P_n -> P_n*B[n][j]*P_j)
    4. Mutable class variables shared across instances
    5. Missing observation_space
    6. Reward zeroed out after computation for intermediate steps

Migrated from OpenAI Gym to Gymnasium (5-tuple step, seed/options reset).
"""

import sys
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from math import sin, exp


class GeneratorsEnv(gym.Env):
    """Gymnasium environment for Combined Economic and Emission Dispatch (CEED).

    The environment models a 10-generator power system over a 24-hour scheduling
    horizon. At each step, the RL agent selects a discrete power level (0-100) for
    one generator at one hour. After all 9 controllable generators are set for an
    hour, the slack generator (unit 1) is computed to balance supply and demand.

    Observation:
        A float32 vector containing:
        - Normalized current hour (m / M)
        - Normalized current generator index ((n - 2) / (N - 1))
        - Power levels of all 10 generators for the current hour (normalized)
        - Current hour demand (normalized)
        - Previous hour total cost (normalized)
        - Previous hour total emissions (normalized)

    Actions:
        Discrete(101) - integer from 0 to 100 representing percentage of the
        generator's power range (0 = p_min, 100 = p_max).

    Reward:
        Negative weighted sum of fuel cost, emissions, and constraint penalties.
        Dense rewards are provided at each generator step.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    # System constants
    M = 24      # Number of hours in a day
    N = 10      # Number of generators (1 slack + 9 agent-controlled)
    E = 10      # Emissions scaling factor
    Wc = 0.225  # Cost weight for linear scalarisation
    We = 0.275  # Emissions weight for linear scalarisation
    Wp = 0.5    # Penalty weight for linear scalarisation
    C = 10e6    # Violation penalty constant

    # Generator characteristics DataFrame
    gen_chars = pd.DataFrame(
        [
            (150, 470, 786.7988, 38.5397, 0.1524, 450, 0.041,
             103.3908, -2.4444, 0.0312, 0.5035, 0.0207, 80, 80),
            (135, 470, 451.3251, 46.1591, 0.1058, 600, 0.036,
             103.3908, -2.4444, 0.0312, 0.5035, 0.0207, 80, 80),
            (73, 340, 1049.9977, 40.3965, 0.0280, 320, 0.028,
             300.3910, -4.0695, 0.0509, 0.4968, 0.0202, 80, 80),
            (60, 300, 1243.5311, 38.3055, 0.0354, 260, 0.052,
             300.3910, -4.0695, 0.0509, 0.4968, 0.0202, 50, 50),
            (73, 243, 1658.5696, 36.3278, 0.0211, 280, 0.063,
             320.0006, -3.8132, 0.0344, 0.4972, 0.0200, 50, 50),
            (57, 160, 1356.6592, 38.2704, 0.0179, 310, 0.048,
             320.0006, -3.8132, 0.0344, 0.4972, 0.0200, 50, 50),
            (20, 130, 1450.7045, 36.5104, 0.0121, 300, 0.086,
             330.0056, -3.9023, 0.0465, 0.5163, 0.0214, 30, 30),
            (47, 120, 1450.7045, 36.5104, 0.0121, 340, 0.082,
             330.0056, -3.9023, 0.0465, 0.5163, 0.0214, 30, 30),
            (20, 80, 1455.6056, 39.5804, 0.1090, 270, 0.098,
             350.0056, -3.9524, 0.0465, 0.5475, 0.0234, 30, 30),
            (10, 55, 1469.4026, 40.5407, 0.1295, 380, 0.094,
             360.0012, -3.9864, 0.0470, 0.5475, 0.0234, 30, 30),
        ],
        columns=[
            "p_min_i", "p_max_i", "a_i", "b_i", "c_i", "d_i", "e_i",
            "alpha_i", "beta_i", "gamma_i", "eta_i", "delta_i", "ur_i", "dr_i",
        ],
        index=[f"unit{i}" for i in range(1, 11)],
    )

    # B-coefficient loss matrix (10x10)
    B = pd.DataFrame(
        [
            (0.000049, 0.000014, 0.000015, 0.000015, 0.000016,
             0.000017, 0.000017, 0.000018, 0.000019, 0.000020),
            (0.000014, 0.000045, 0.000016, 0.000016, 0.000017,
             0.000015, 0.000015, 0.000016, 0.000018, 0.000018),
            (0.000015, 0.000016, 0.000039, 0.000010, 0.000012,
             0.000012, 0.000014, 0.000014, 0.000016, 0.000016),
            (0.000015, 0.000016, 0.000010, 0.000040, 0.000014,
             0.000010, 0.000011, 0.000012, 0.000014, 0.000015),
            (0.000016, 0.000017, 0.000012, 0.000014, 0.000035,
             0.000011, 0.000013, 0.000013, 0.000015, 0.000016),
            (0.000017, 0.000015, 0.000012, 0.000010, 0.000011,
             0.000036, 0.000012, 0.000012, 0.000014, 0.000015),
            (0.000017, 0.000015, 0.000014, 0.000011, 0.000013,
             0.000012, 0.000038, 0.000016, 0.000016, 0.000018),
            (0.000018, 0.000016, 0.000014, 0.000012, 0.000013,
             0.000012, 0.000016, 0.000040, 0.000015, 0.000016),
            (0.000019, 0.000018, 0.000016, 0.000014, 0.000015,
             0.000014, 0.000016, 0.000015, 0.000042, 0.000019),
            (0.000020, 0.000018, 0.000016, 0.000015, 0.000016,
             0.000015, 0.000018, 0.000016, 0.000019, 0.000044),
        ]
    )

    # Hourly power demand (MW)
    hour_power_demand = pd.DataFrame(
        [1036, 1110, 1258, 1406, 1480, 1628, 1702, 1776, 1924, 2022,
         2106, 2150, 2072, 1924, 1776, 1554, 1480, 1628, 1776, 1972,
         1924, 1628, 1332, 1184],
        columns=["p_d_n"],
        index=[f"hour{i}" for i in range(1, 25)],
    )

    hour_power_demand_diff = hour_power_demand.diff().fillna(0.0)
    hour_power_demand_diff.rename(columns={"p_d_n": "delta_p_d"}, inplace=True)

    # Normalisation constants for observation space
    _MAX_POWER = 470.0   # Largest p_max across all generators
    _MAX_DEMAND = 2150.0  # Peak demand
    _MAX_COST = 100000.0  # Approximate max hourly cost for normalisation
    _MAX_EMISSIONS = 100000.0  # Approximate max hourly emissions

    def __init__(self, render_mode=None):
        """Initialise the Generators environment.

        Args:
            render_mode: Optional render mode (only 'human' supported).
        """
        super().__init__()
        self.render_mode = render_mode

        # BUG FIX 4: Instance-level mutable state (not class-level)
        self.states_array = np.zeros([self.M * self.N, 2])
        self.p_n_m_df = pd.DataFrame(
            0.0,
            columns=[f"hour{i}" for i in range(1, 25)],
            index=[f"unit{i}" for i in range(1, 11)],
        )

        # Action space: discrete 0-100 (percentage of generator power range)
        self.action_space = spaces.Discrete(101)

        # BUG FIX 5: Define observation_space
        # Observation vector: [hour_norm, gen_norm, 10 gen powers, demand, prev_cost, prev_emissions]
        obs_dim = 2 + self.N + 3  # = 15
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Episode state
        self.m = 1       # Current hour (1-indexed)
        self.n = 2       # Current generator (1-indexed, start at 2; 1 is slack)
        self.reward = 0.0
        self.done = False
        self.prev_cost = 0.0
        self.prev_emissions = 0.0

    def _init_states_array(self):
        """Initialise the states_array with random starting power levels and demand diffs."""
        self.states_array.fill(0.0)
        for n in range(1, self.N + 1):
            idx = (n - 1) * self.M
            self.states_array[idx] = [0.0, self._random_power(n)]
        # Fill demand diffs
        m_idx = 0
        for state_idx in range(len(self.states_array)):
            self.states_array[state_idx][0] = self.hour_power_demand_diff.iloc[m_idx, 0]
            m_idx += 1
            if m_idx == 24:
                m_idx = 0

    def _random_power(self, n):
        """Generate a random power output for generator n within its bounds.

        Args:
            n: Generator number (1-indexed).

        Returns:
            Random power value between p_min and p_max for generator n.
        """
        assert isinstance(n, int) and 1 <= n <= self.N
        p_min = self.gen_chars.iloc[n - 1, 0]  # p_min_i
        p_max = self.gen_chars.iloc[n - 1, 1]  # p_max_i
        action = self.np_random.integers(0, self.action_space.n)
        return p_min + (p_max - p_min) * (action / 100.0)

    def _build_observation(self):
        """Build the observation vector for the current state.

        Returns:
            numpy.ndarray of shape (obs_dim,) with float32 dtype.
        """
        hour_norm = (self.m - 1) / max(self.M - 1, 1)
        gen_norm = (self.n - 2) / max(self.N - 2, 1)

        # Power levels of all generators for current hour (normalised)
        gen_powers = np.zeros(self.N, dtype=np.float32)
        for i in range(self.N):
            gen_powers[i] = self.p_n_m_df.iloc[i, self.m - 1] / self._MAX_POWER

        demand_norm = self.get_p_d_m(self.m) / self._MAX_DEMAND
        cost_norm = np.clip(self.prev_cost / self._MAX_COST, -1.0, 1.0)
        emis_norm = np.clip(self.prev_emissions / self._MAX_EMISSIONS, -1.0, 1.0)

        obs = np.array(
            [hour_norm, gen_norm] + gen_powers.tolist() + [demand_norm, cost_norm, emis_norm],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0)

    def set_p_n_m_a(self, n, m, action):
        """Set power output for generator n at hour m based on a discrete action.

        Args:
            n: Generator number (2-10, 1-indexed).
            m: Hour (1-24, 1-indexed).
            action: Discrete action in [0, 100].

        Returns:
            The power level set for the generator (MW).
        """
        assert isinstance(n, int) and 2 <= n <= self.N
        assert isinstance(m, int) and 1 <= m <= self.M
        assert self.action_space.contains(action)

        p_min = self.gen_chars.iloc[n - 1, 0]
        p_max = self.gen_chars.iloc[n - 1, 1]
        p_n_m_a = p_min + (p_max - p_min) * (action / 100.0)

        self.p_n_m_df.iloc[n - 1, m - 1] = p_n_m_a

        # BUG FIX 2: Consistent indexing formula: (n-1)*M + (m-1) for current hour
        self.states_array[((n - 1) * self.M) + (m - 1)][1] = p_n_m_a
        # Also store in next slot for prev-hour lookups if valid
        if m < self.M:
            self.states_array[((n - 1) * self.M) + m][1] = p_n_m_a
        return p_n_m_a

    def get_p_d_m(self, m):
        """Get power demand at hour m.

        Args:
            m: Hour (1-24, 1-indexed).

        Returns:
            Power demand in MW.
        """
        assert isinstance(m, int) and 1 <= m <= self.M
        return self.hour_power_demand.iloc[m - 1, 0]

    def get_p_n_m(self, n, m):
        """Get power output of generator n at hour m.

        Args:
            n: Generator number (1-10).
            m: Hour (1-24).

        Returns:
            Power output in MW.
        """
        if isinstance(n, str):
            n = self.p_n_m_df.index.get_loc(n) + 1
        if isinstance(m, str):
            m = self.p_n_m_df.columns.get_loc(m) + 1
        assert isinstance(n, int) and 1 <= n <= self.N
        assert isinstance(m, int) and 1 <= m <= self.M
        return self.p_n_m_df.iloc[n - 1, m - 1]

    def get_p_n_m_prev(self, n, m):
        """Get power output of generator n at the previous hour (for ramp constraints).

        For hour 1, returns the initial random power stored in states_array.

        Args:
            n: Generator number (1-10).
            m: Current hour (1-24).

        Returns:
            Power output from previous hour in MW.
        """
        if isinstance(n, str):
            n = self.p_n_m_df.index.get_loc(n) + 1
        if isinstance(m, str):
            m = self.p_n_m_df.columns.get_loc(m) + 1
        assert isinstance(n, int) and 1 <= n <= self.N
        assert isinstance(m, int) and 1 <= m <= self.M

        if m == 1:
            # Return initial power from states_array
            return self.states_array[((n - 1) * self.M)][1]
        else:
            return self.p_n_m_df.iloc[n - 1, m - 2]

    def get_f_c_l(self, n, m):
        """Calculate fuel cost for generator n at hour m.

        Uses the valve-point cost function:
            F_c = a + b*P + c*P^2 + |d*sin(e*(P_min - P))|

        Args:
            n: Generator number (1-10).
            m: Hour (1-24).

        Returns:
            Fuel cost in $/h.
        """
        assert isinstance(n, int) and 1 <= n <= self.N
        assert isinstance(m, int) and 1 <= m <= self.M

        gc = self.gen_chars.iloc[n - 1]
        a_n = gc["a_i"]
        b_n = gc["b_i"]
        c_n = gc["c_i"]
        d_n = gc["d_i"]
        e_n = gc["e_i"]
        p_min_n = gc["p_min_i"]
        p_n_m = self.get_p_n_m(n, m)

        return a_n + b_n * p_n_m + c_n * p_n_m**2 + abs(d_n * sin(e_n * (p_min_n - p_n_m)))

    def get_f_c_g(self, m):
        """Calculate total fuel cost across all generators at hour m.

        Args:
            m: Hour (1-24).

        Returns:
            Total fuel cost in $/h.
        """
        assert isinstance(m, int) and 1 <= m <= self.M
        return sum(self.get_f_c_l(n, m) for n in range(1, self.N + 1))

    def get_f_e_l(self, n, m):
        """Calculate emissions for generator n at hour m.

        Uses the emission function:
            F_e = E * (alpha + beta*P + gamma*P^2 + eta*exp(delta*P))

        Args:
            n: Generator number (1-10).
            m: Hour (1-24).

        Returns:
            Emissions (scaled by E).
        """
        assert isinstance(n, int) and 1 <= n <= self.N
        assert isinstance(m, int) and 1 <= m <= self.M

        gc = self.gen_chars.iloc[n - 1]
        alpha_n = gc["alpha_i"]
        beta_n = gc["beta_i"]
        gamma_n = gc["gamma_i"]
        eta_n = gc["eta_i"]
        delta_n = gc["delta_i"]
        p_n_m = self.get_p_n_m(n, m)

        return self.E * (alpha_n + beta_n * p_n_m + gamma_n * p_n_m**2 + eta_n * exp(delta_n * p_n_m))

    def get_f_e_g(self, m):
        """Calculate total emissions across all generators at hour m.

        Args:
            m: Hour (1-24).

        Returns:
            Total emissions.
        """
        return sum(self.get_f_e_l(n, m) for n in range(1, self.N + 1))

    def get_p_1_m(self, m):
        """Calculate the slack generator (unit 1) power output at hour m.

        Solves the quadratic power balance equation:
            P_d + P_loss = sum(P_n)
        where P_loss depends on all generator outputs via the B-matrix.

        BUG FIX 3: The inner loop now correctly uses P_j (not P_n) in the
        B-matrix product: sum_c1 += P_n * B[n][j] * P_j

        Args:
            m: Hour (1-24).

        Returns:
            Slack generator power output in MW.
        """
        assert 1 <= m <= self.M

        sum_b = 0.0
        sum_c1 = 0.0
        sum_c2 = 0.0

        for n in range(2, self.N + 1):
            sum_b += self.B[0][n - 1] * self.get_p_n_m(n, m)

        for n in range(2, self.N + 1):
            P_n = self.get_p_n_m(n, m)
            for j in range(2, self.N + 1):
                # BUG FIX 3: Use P_j, not P_n for the second factor
                P_j = self.get_p_n_m(j, m)
                sum_c1 += P_n * self.B[n - 1][j - 1] * P_j

        for n in range(2, self.N + 1):
            sum_c2 += self.get_p_n_m(n, m)

        # Quadratic coefficients: a*P1^2 + b*P1 + c = 0
        a = self.B[0][0]
        b = (2 * sum_b) - 1
        c = self.get_p_d_m(m) + sum_c1 - sum_c2

        d = b**2 - 4 * a * c
        if d < 0:
            # Fallback: return demand minus other generators (ignore losses)
            p_1_m = self.get_p_d_m(m) - sum_c2
        elif d == 0:
            p_1_m = -b / (2 * a)
        else:
            p_1_m = (-b - math.sqrt(d)) / (2 * a)

        # Store in dataframe and states_array
        self.p_n_m_df.iloc[0, m - 1] = p_1_m
        self.states_array[m - 1][1] = p_1_m
        return p_1_m

    def get_f_p_g(self, m):
        """Calculate global penalty for constraint violations at hour m.

        Checks capacity and ramp-rate constraints for the slack generator (unit 1).

        Args:
            m: Hour (1-24).

        Returns:
            Penalty value (0 if no violations).
        """
        p_1_m = self.get_p_n_m(1, m)
        p_1_max = self.gen_chars.loc["unit1", "p_max_i"]
        p_1_min = self.gen_chars.loc["unit1", "p_min_i"]
        p_1_m_prev = self.get_p_n_m_prev(1, m)
        ur1 = self.gen_chars.loc["unit1", "ur_i"]
        dr1 = self.gen_chars.loc["unit1", "dr_i"]

        # Capacity constraint
        h1 = 0.0
        delta1 = 0
        if p_1_m > p_1_max:
            h1 = p_1_m - p_1_max
            delta1 = 1
        elif p_1_m < p_1_min:
            h1 = p_1_min - p_1_m
            delta1 = 1

        # Ramp-rate constraint
        h2 = 0.0
        delta2 = 0
        ramp = p_1_m - p_1_m_prev
        if ramp > ur1:
            h2 = ramp - ur1
            delta2 = 1
        elif ramp < -dr1:
            h2 = ramp + dr1
            delta2 = 1

        return (self.C * abs(h1 + 1) * delta1) + (self.C * abs(h2 + 1) * delta2)

    def find_constrained_action_space(self, unit, p_n_m_prev):
        """Find the set of valid actions respecting ramp-rate constraints.

        Args:
            unit: Generator number (1-indexed).
            p_n_m_prev: Power output of this generator in the previous hour.

        Returns:
            numpy.ndarray of valid action indices.
        """
        p_min_n = self.gen_chars.iloc[unit - 1, 0]
        p_max_n = self.gen_chars.iloc[unit - 1, 1]
        dr_n = self.gen_chars.iloc[unit - 1, self.gen_chars.columns.get_loc("dr_i")]
        ur_n = self.gen_chars.iloc[unit - 1, self.gen_chars.columns.get_loc("ur_i")]

        num_segments = self.action_space.n - 1  # 100
        power_per_segment = (p_max_n - p_min_n) / num_segments

        if power_per_segment == 0:
            return np.array([0])

        current_action = int(round((p_n_m_prev - p_min_n) / power_per_segment))
        current_action = np.clip(current_action, 0, num_segments)

        max_down = int(round(dr_n / power_per_segment))
        max_up = int(round(ur_n / power_per_segment))

        lower = max(0, current_action - max_down)
        upper = min(num_segments, current_action + max_up)

        return np.arange(lower, upper + 1)

    def step(self, action):
        """Execute one step in the environment.

        The agent sets one generator's power level. After all 9 controllable
        generators are set for an hour, the slack generator is computed and
        the hour advances.

        Args:
            action: Integer in [0, 100].

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        action = int(action)
        terminated = False
        truncated = False
        info = {}

        constrained_actions = self.find_constrained_action_space(
            self.n, self.get_p_n_m_prev(self.n, self.m)
        )

        if action in constrained_actions:
            self.set_p_n_m_a(self.n, self.m, action)
            # Dense reward: per-generator cost and emissions feedback
            rc = self.get_f_c_l(self.n, self.m)
            re = self.get_f_e_l(self.n, self.m)
            rp = 0.0
            info["action_valid"] = True
        else:
            # Invalid action: copy previous hour's power (or keep at init)
            prev_power = self.get_p_n_m_prev(self.n, self.m)
            self.p_n_m_df.iloc[self.n - 1, self.m - 1] = prev_power
            # BUG FIX 2: Use consistent indexing
            self.states_array[((self.n - 1) * self.M) + (self.m - 1)][1] = prev_power
            rc = 1000000.0
            re = 1000000.0
            rp = 1000000.0
            info["action_valid"] = False

        self.n += 1

        # BUG FIX 6: Compute dense reward BEFORE any overwrite
        reward = -(self.Wc * rc + self.We * re + self.Wp * rp)

        # BUG FIX 1: Check termination BEFORE resetting n
        if self.n == 11:
            # All controllable generators set for this hour
            self.get_p_1_m(self.m)  # Compute slack generator

            # Full-hour cooperative reward replaces dense per-gen reward
            rc = self.get_f_c_g(self.m)
            re = self.get_f_e_g(self.m)
            rp = self.get_f_p_g(self.m)
            reward = -(self.Wc * rc + self.We * re + self.Wp * rp)

            self.prev_cost = rc
            self.prev_emissions = re

            # Populate info with diagnostics
            info["hour"] = self.m
            info["hourly_cost"] = rc
            info["hourly_emissions"] = re
            info["hourly_penalty"] = rp
            info["slack_power"] = self.get_p_n_m(1, self.m)
            info["gen_powers"] = [
                self.get_p_n_m(i, self.m) for i in range(1, self.N + 1)
            ]
            info["demand"] = self.get_p_d_m(self.m)

            # Check termination: last generator of last hour
            if self.m == self.M:
                terminated = True
                self.done = True
                obs = self._build_observation()
                return obs, reward, terminated, truncated, info

            # Advance to next hour
            self.m += 1
            self.n = 2

        self.reward = reward
        obs = self._build_observation()
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state.

        Args:
            seed: Optional random seed.
            options: Optional dict of reset options.

        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed, options=options)

        self.m = 1
        self.n = 2
        self.done = False
        self.reward = 0.0
        self.prev_cost = 0.0
        self.prev_emissions = 0.0

        # BUG FIX 4: Reset instance-level mutable state
        self.p_n_m_df = pd.DataFrame(
            0.0,
            columns=[f"hour{i}" for i in range(1, 25)],
            index=[f"unit{i}" for i in range(1, 11)],
        )
        self._init_states_array()

        # Set initial random power levels for hour 1 into p_n_m_df
        for n in range(1, self.N + 1):
            init_power = self.states_array[(n - 1) * self.M][1]
            self.p_n_m_df.iloc[n - 1, 0] = init_power

        obs = self._build_observation()
        info = {"hour": self.m, "generator": self.n, "demand": self.get_p_d_m(self.m)}
        return obs, info

    def render(self):
        """Render the current environment state."""
        if self.render_mode == "human":
            print(f"Hour: {self.m}/{self.M}, Generator: {self.n}/{self.N}")
            print(f"Demand: {self.get_p_d_m(self.m)} MW")
            for i in range(1, self.N + 1):
                print(f"  Unit {i}: {self.p_n_m_df.iloc[i-1, self.m-1]:.1f} MW")

    def close(self):
        """Clean up resources."""
        pass
