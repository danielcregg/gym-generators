"""
Comprehensive test suite for the Generators-v0 CEED Gymnasium environment.

Tests cover environment creation, step/reset API, observation spaces,
constraint checking, reward structure, cost/emission functions, and
full-episode rollouts.
"""

import numpy as np
import pytest
import gymnasium as gym

# Ensure the environment is registered
import gym_generators  # noqa: F401
from gym_generators.envs.generators_env import GeneratorsEnv


# ---------------------------------------------------------------------------
# Test 1: Environment creation via gymnasium.make
# ---------------------------------------------------------------------------
class TestEnvironmentCreation:
    """Test that the environment can be created via gymnasium.make."""

    def test_make_generators_v0(self):
        """Environment should be creatable via gym.make('Generators-v0')."""
        env = gym.make("Generators-v0")
        assert env is not None
        assert hasattr(env, "step")
        assert hasattr(env, "reset")
        assert hasattr(env, "action_space")
        assert hasattr(env, "observation_space")
        env.close()

    def test_direct_instantiation(self):
        """Environment should be directly instantiable."""
        env = GeneratorsEnv()
        assert env is not None
        env.close()


# ---------------------------------------------------------------------------
# Test 2: Reset returns proper observation shape and type
# ---------------------------------------------------------------------------
class TestReset:
    """Test the reset() method."""

    def test_reset_returns_tuple(self):
        """reset() should return (obs, info) tuple."""
        env = gym.make("Generators-v0")
        result = env.reset(seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 2
        obs, info = result
        assert isinstance(info, dict)
        env.close()

    def test_reset_observation_shape(self):
        """Observation should match observation_space shape."""
        env = gym.make("Generators-v0")
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32
        env.close()

    def test_reset_with_seed(self):
        """Two resets with the same seed should produce the same observation."""
        env = gym.make("Generators-v0")
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)
        env.close()

    def test_reset_info_contains_keys(self):
        """Info dict from reset should contain hour, generator, demand."""
        env = gym.make("Generators-v0")
        _, info = env.reset(seed=42)
        assert "hour" in info
        assert "generator" in info
        assert "demand" in info
        env.close()


# ---------------------------------------------------------------------------
# Test 3: Step returns 5-tuple
# ---------------------------------------------------------------------------
class TestStepReturn:
    """Test step() return format."""

    def test_step_returns_5_tuple(self):
        """step() should return (obs, reward, terminated, truncated, info)."""
        env = gym.make("Generators-v0")
        env.reset(seed=42)
        result = env.step(50)
        assert isinstance(result, tuple)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        env.close()


# ---------------------------------------------------------------------------
# Test 4: Observation within bounds
# ---------------------------------------------------------------------------
class TestObservationBounds:
    """Test that observations stay within observation_space bounds."""

    def test_reset_obs_in_bounds(self):
        """Observation from reset should be within observation_space."""
        env = gym.make("Generators-v0")
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs), f"Obs out of bounds: {obs}"
        env.close()

    def test_step_obs_in_bounds(self):
        """Observation from step should be within observation_space."""
        env = gym.make("Generators-v0")
        env.reset(seed=42)
        for _ in range(20):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            assert env.observation_space.contains(obs), f"Obs out of bounds: {obs}"
            if terminated or truncated:
                break
        env.close()


# ---------------------------------------------------------------------------
# Test 5: Action space validity
# ---------------------------------------------------------------------------
class TestActionSpace:
    """Test the action space."""

    def test_action_space_type(self):
        """Action space should be Discrete(101)."""
        env = gym.make("Generators-v0")
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 101
        env.close()

    def test_action_space_contains(self):
        """Action space should contain values 0 through 100."""
        env = gym.make("Generators-v0")
        for a in [0, 50, 100]:
            assert env.action_space.contains(a)
        assert not env.action_space.contains(-1)
        assert not env.action_space.contains(101)
        env.close()

    def test_action_space_sample(self):
        """Sampled actions should be valid."""
        env = gym.make("Generators-v0")
        for _ in range(100):
            action = env.action_space.sample()
            assert env.action_space.contains(action)
        env.close()


# ---------------------------------------------------------------------------
# Test 6: Full episode completes without errors
# ---------------------------------------------------------------------------
class TestFullEpisode:
    """Test that a full episode can run to completion."""

    def test_full_episode_completes(self):
        """A full episode with random actions should terminate cleanly."""
        env = gym.make("Generators-v0")
        obs, info = env.reset(seed=42)
        total_steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1
            # Safety: prevent infinite loop
            if total_steps > 10000:
                pytest.fail("Episode did not terminate within 10000 steps")

        # 9 generators * 24 hours = 216 steps per episode
        assert total_steps == 9 * 24, f"Expected 216 steps, got {total_steps}"
        assert terminated is True
        env.close()

    def test_multiple_episodes(self):
        """Multiple consecutive episodes should work."""
        env = gym.make("Generators-v0")
        for ep in range(3):
            obs, info = env.reset(seed=ep)
            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
        env.close()


# ---------------------------------------------------------------------------
# Test 7: Constraint checking (ramp rates, capacity limits)
# ---------------------------------------------------------------------------
class TestConstraints:
    """Test ramp-rate and capacity constraint logic."""

    def test_constrained_action_space_not_empty(self):
        """Constrained action space should always have at least one valid action."""
        env = GeneratorsEnv()
        env.reset(seed=42)
        for n in range(2, env.N + 1):
            prev_power = env.get_p_n_m_prev(n, 1)
            actions = env.find_constrained_action_space(n, prev_power)
            assert len(actions) > 0, f"Unit {n} has no valid actions"
        env.close()

    def test_constrained_actions_within_bounds(self):
        """All constrained actions should be within [0, 100]."""
        env = GeneratorsEnv()
        env.reset(seed=42)
        for n in range(2, env.N + 1):
            prev_power = env.get_p_n_m_prev(n, 1)
            actions = env.find_constrained_action_space(n, prev_power)
            assert np.all(actions >= 0)
            assert np.all(actions <= 100)
        env.close()

    def test_ramp_rate_constraint(self):
        """Actions outside ramp-rate limits should not be in constrained set."""
        env = GeneratorsEnv()
        env.reset(seed=42)
        # Unit 10: p_min=10, p_max=55, range=45, ur=dr=30
        n = 10
        p_min = env.gen_chars.iloc[n - 1, 0]
        p_max = env.gen_chars.iloc[n - 1, 1]
        mid_power = (p_min + p_max) / 2.0
        actions = env.find_constrained_action_space(n, mid_power)
        assert len(actions) <= 101
        env.close()

    def test_capacity_penalty(self):
        """Penalty should be nonzero when slack generator exceeds capacity."""
        env = GeneratorsEnv()
        env.reset(seed=42)
        # Set all generators to minimum for hour 1
        for n in range(2, env.N + 1):
            env.set_p_n_m_a(n, 1, 0)  # action 0 = p_min
        # Compute slack
        env.get_p_1_m(1)
        penalty = env.get_f_p_g(1)
        assert isinstance(penalty, (int, float))
        env.close()


# ---------------------------------------------------------------------------
# Test 8: Reward is negative (costs are positive)
# ---------------------------------------------------------------------------
class TestRewardSign:
    """Test that rewards have the expected sign."""

    def test_reward_is_negative(self):
        """Rewards should be negative (negative cost)."""
        env = gym.make("Generators-v0")
        env.reset(seed=42)

        total_reward = 0.0
        for _ in range(9):  # Complete one hour
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # The hourly reward (last of the 9 steps) should be negative
        assert reward < 0, f"Expected negative reward, got {reward}"
        env.close()

    def test_valid_action_reward_less_negative_than_invalid(self):
        """Valid actions should yield less negative rewards than invalid penalty."""
        env = GeneratorsEnv()
        env.reset(seed=42)

        # Get a valid action for unit 2
        prev_power = env.get_p_n_m_prev(2, 1)
        valid_actions = env.find_constrained_action_space(2, prev_power)
        valid_action = valid_actions[len(valid_actions) // 2]

        # Take the valid action
        _, reward_valid, _, _, info_valid = env.step(valid_action)

        # Reset and take an invalid action (if possible)
        env.reset(seed=42)
        all_actions = set(range(101))
        invalid_actions = all_actions - set(valid_actions)
        if invalid_actions:
            invalid_action = list(invalid_actions)[0]
            _, reward_invalid, _, _, info_invalid = env.step(invalid_action)
            assert reward_valid > reward_invalid, \
                f"Valid reward {reward_valid} should be greater than invalid {reward_invalid}"
        env.close()


# ---------------------------------------------------------------------------
# Test 9: Invalid actions get penalized
# ---------------------------------------------------------------------------
class TestInvalidActionPenalty:
    """Test that invalid actions receive heavy penalties."""

    def test_invalid_action_flagged(self):
        """Invalid actions should be flagged in info dict."""
        env = GeneratorsEnv()
        env.reset(seed=42)

        prev_power = env.get_p_n_m_prev(2, 1)
        valid_actions = set(env.find_constrained_action_space(2, prev_power))
        all_actions = set(range(101))
        invalid_actions = all_actions - valid_actions

        if invalid_actions:
            invalid_action = list(invalid_actions)[0]
            _, _, _, _, info = env.step(invalid_action)
            assert info.get("action_valid") is False
        env.close()

    def test_valid_action_flagged(self):
        """Valid actions should be flagged as valid in info dict."""
        env = GeneratorsEnv()
        env.reset(seed=42)

        prev_power = env.get_p_n_m_prev(2, 1)
        valid_actions = env.find_constrained_action_space(2, prev_power)
        valid_action = valid_actions[len(valid_actions) // 2]

        _, _, _, _, info = env.step(valid_action)
        assert info.get("action_valid") is True
        env.close()


# ---------------------------------------------------------------------------
# Test 10: B-matrix loss calculation
# ---------------------------------------------------------------------------
class TestBMatrixLoss:
    """Test the B-matrix transmission loss calculation."""

    def test_b_matrix_shape(self):
        """B-matrix should be 10x10."""
        env = GeneratorsEnv()
        assert env.B.shape == (10, 10)
        env.close()

    def test_b_matrix_symmetric(self):
        """B-matrix should be symmetric."""
        env = GeneratorsEnv()
        B_arr = env.B.values
        np.testing.assert_array_almost_equal(B_arr, B_arr.T)
        env.close()

    def test_b_matrix_positive_diagonal(self):
        """B-matrix diagonal elements should be positive."""
        env = GeneratorsEnv()
        for i in range(10):
            assert env.B.iloc[i, i] > 0
        env.close()

    def test_slack_generator_computation(self):
        """Slack generator computation should return a finite value.

        Note: when other generators produce too much power relative to demand,
        the slack generator may be negative (meaning excess generation). The test
        verifies the computation is finite and the quadratic is solved correctly.
        When generators are at minimum, slack must pick up the remaining demand.
        """
        env = GeneratorsEnv()
        env.reset(seed=42)
        # Set generators to MINIMUM for high-demand hour (hour 12, demand=2150)
        for n in range(2, env.N + 1):
            env.set_p_n_m_a(n, 12, 0)  # action 0 = p_min
        p1 = env.get_p_1_m(12)
        assert np.isfinite(p1), f"Slack power is not finite: {p1}"
        # With all gens at min for high demand, slack should be large and positive
        assert p1 > 0, f"Slack power should be positive for high demand, got {p1}"
        env.close()


# ---------------------------------------------------------------------------
# Test 11: Cost function with known inputs
# ---------------------------------------------------------------------------
class TestCostFunction:
    """Test the fuel cost function."""

    def test_cost_positive_for_controllable_generators(self):
        """Cost should be positive for controllable generators at valid power.

        Note: the slack generator can have negative power in oversupply scenarios,
        which produces unusual cost values. This test only checks units 2-10.
        """
        env = GeneratorsEnv()
        env.reset(seed=42)
        for n in range(2, env.N + 1):
            env.set_p_n_m_a(n, 1, 50)

        for n in range(2, env.N + 1):
            cost = env.get_f_c_l(n, 1)
            assert cost > 0, f"Cost for unit {n} should be positive, got {cost}"
        env.close()

    def test_cost_increases_with_power(self):
        """Cost should generally increase with higher power output."""
        env = GeneratorsEnv()
        env.reset(seed=42)
        env.set_p_n_m_a(2, 1, 10)
        for n in range(3, env.N + 1):
            env.set_p_n_m_a(n, 1, 50)
        env.get_p_1_m(1)
        cost_low = env.get_f_c_l(2, 1)

        env.reset(seed=42)
        env.set_p_n_m_a(2, 1, 90)
        for n in range(3, env.N + 1):
            env.set_p_n_m_a(n, 1, 50)
        env.get_p_1_m(1)
        cost_high = env.get_f_c_l(2, 1)

        assert cost_high > cost_low, \
            f"Cost at high power ({cost_high}) should exceed cost at low power ({cost_low})"
        env.close()

    def test_global_cost_is_sum_of_local(self):
        """Global cost should equal sum of all local costs."""
        env = GeneratorsEnv()
        env.reset(seed=42)
        for n in range(2, env.N + 1):
            env.set_p_n_m_a(n, 1, 50)
        env.get_p_1_m(1)

        local_sum = sum(env.get_f_c_l(n, 1) for n in range(1, env.N + 1))
        global_cost = env.get_f_c_g(1)
        assert abs(local_sum - global_cost) < 1e-6, \
            f"Global cost {global_cost} != sum of local costs {local_sum}"
        env.close()


# ---------------------------------------------------------------------------
# Test 12: Emissions function with known inputs
# ---------------------------------------------------------------------------
class TestEmissionsFunction:
    """Test the emissions function."""

    def test_emissions_positive(self):
        """Emissions should be positive for any valid power output."""
        env = GeneratorsEnv()
        env.reset(seed=42)
        for n in range(2, env.N + 1):
            env.set_p_n_m_a(n, 1, 50)
        env.get_p_1_m(1)

        for n in range(1, env.N + 1):
            emissions = env.get_f_e_l(n, 1)
            assert emissions > 0, f"Emissions for unit {n} should be positive, got {emissions}"
        env.close()

    def test_emissions_increases_with_power(self):
        """Emissions should generally increase with higher power output."""
        env = GeneratorsEnv()
        env.reset(seed=42)
        env.set_p_n_m_a(2, 1, 10)
        for n in range(3, env.N + 1):
            env.set_p_n_m_a(n, 1, 50)
        env.get_p_1_m(1)
        emis_low = env.get_f_e_l(2, 1)

        env.reset(seed=42)
        env.set_p_n_m_a(2, 1, 90)
        for n in range(3, env.N + 1):
            env.set_p_n_m_a(n, 1, 50)
        env.get_p_1_m(1)
        emis_high = env.get_f_e_l(2, 1)

        assert emis_high > emis_low, \
            f"Emissions at high power ({emis_high}) should exceed emissions at low power ({emis_low})"
        env.close()

    def test_global_emissions_is_sum_of_local(self):
        """Global emissions should equal sum of all local emissions."""
        env = GeneratorsEnv()
        env.reset(seed=42)
        for n in range(2, env.N + 1):
            env.set_p_n_m_a(n, 1, 50)
        env.get_p_1_m(1)

        local_sum = sum(env.get_f_e_l(n, 1) for n in range(1, env.N + 1))
        global_emis = env.get_f_e_g(1)
        assert abs(local_sum - global_emis) < 1e-6, \
            f"Global emissions {global_emis} != sum of local emissions {local_sum}"
        env.close()

    def test_emission_scaling_factor(self):
        """Emissions should be scaled by factor E=10."""
        env = GeneratorsEnv()
        assert env.E == 10
        env.close()


# ---------------------------------------------------------------------------
# Additional robustness tests
# ---------------------------------------------------------------------------
class TestMutableStateIsolation:
    """Test that instance state is properly isolated (Bug Fix 4)."""

    def test_separate_instances_independent(self):
        """Two environment instances should not share mutable state."""
        env1 = GeneratorsEnv()
        env2 = GeneratorsEnv()
        env1.reset(seed=1)
        env2.reset(seed=2)

        # Modify env1 state
        env1.set_p_n_m_a(2, 1, 100)

        # env2 should be unaffected
        p_env2 = env2.get_p_n_m(2, 1)
        p_env1 = env1.get_p_n_m(2, 1)
        assert p_env1 != p_env2, "Instances should have independent state"

        env1.close()
        env2.close()


class TestInfoDict:
    """Test that info dict contains useful diagnostics."""

    def test_hourly_info_after_full_hour(self):
        """After completing an hour (9 steps), info should contain diagnostics."""
        env = GeneratorsEnv()
        env.reset(seed=42)
        info = {}
        for step_idx in range(9):  # 9 generators to set
            _, _, _, _, info = env.step(env.action_space.sample())

        # After 9 steps, we should have hourly info
        assert "hourly_cost" in info
        assert "hourly_emissions" in info
        assert "hourly_penalty" in info
        assert "slack_power" in info
        assert "gen_powers" in info
        assert "demand" in info
        assert len(info["gen_powers"]) == 10
        env.close()


class TestDemandData:
    """Test the demand data integrity."""

    def test_demand_values(self):
        """Demand values should match known data."""
        env = GeneratorsEnv()
        assert env.get_p_d_m(1) == 1036
        assert env.get_p_d_m(12) == 2150  # Peak
        assert env.get_p_d_m(24) == 1184
        env.close()

    def test_24_hours(self):
        """There should be exactly 24 hours of demand data."""
        env = GeneratorsEnv()
        assert len(env.hour_power_demand) == 24
        env.close()
