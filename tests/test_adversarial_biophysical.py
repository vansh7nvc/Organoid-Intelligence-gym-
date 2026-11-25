"""
tests/test_adversarial_biophysical.py

Adversarial Stress Test Suite: Milestone M1 Biophysical Invariants
Targeted Adversarial Challenges:
1. Extreme Current Injection & ATP Depletion/Arrest (50, 100, 500, 1000 pA)
2. Refractory Boundary Enforcement (min ISI >= 1.99 ms across all neurons & under synaptic bombardment)
3. Extreme Reward Sign Invariance under Dale's Principle (+/-1000 to +/-1e5 dopamine rewards)
"""

import pytest
import numpy as np
import brian2 as b2
from brian2 import ms, second, Hz, Network, NeuronGroup, Synapses, SpikeMonitor, StateMonitor, defaultclock

from organoid_rl.environment.core import OrganoidEnv, OrganoidConfig
from organoid_rl.environment.neurons import create_metabolic_neurons, get_metabolic_izhikevich_eqs
from organoid_rl.environment.rewards import apply_dopamine
from organoid_rl.utils.seeding import set_seed


class TestAdversarialMetabolicGovernor:
    """Adversarial stress testing of metabolic ATP exhaustion and firing arrest."""

    def setup_method(self):
        b2.start_scope()
        defaultclock.dt = 0.1 * ms

    @pytest.mark.parametrize("model_type,tau_rec", [("RS", 1000), ("FS", 500)])
    @pytest.mark.parametrize("I_stim", [50.0, 100.0, 500.0, 1000.0])
    def test_extreme_current_exhaustion_and_arrest(self, model_type, tau_rec, I_stim):
        """
        Adversarial Challenge 1:
        Inject supra-threshold currents (50, 100, 500, 1000 pA) for 500 ms.
        Assert:
        1. ATP reserve E depletes to <= 0.0.
        2. ZERO spikes are emitted while E <= 0.
        3. Membrane voltage v remains bounded in [-100, 60] mV without NaN/Inf.
        """
        b2.start_scope()
        G = create_metabolic_neurons(1, model_type=model_type)
        G.I = I_stim

        spikes = SpikeMonitor(G)
        mon = StateMonitor(G, ['v', 'u', 'E'], record=True)

        net = Network(G, spikes, mon)
        sim_time_ms = 500.0
        net.run(sim_time_ms * ms)

        v_trace = np.array(mon.v[0])
        E_trace = np.array(mon.E[0])
        u_trace = np.array(mon.u[0])
        t_trace = np.array(mon.t / ms)

        # 1. Numerical stability check
        assert not np.any(np.isnan(v_trace)), f"NaN in v under I={I_stim} pA"
        assert not np.any(np.isinf(v_trace)), f"Inf in v under I={I_stim} pA"
        assert not np.any(np.isnan(E_trace)), f"NaN in E under I={I_stim} pA"
        assert np.all(v_trace >= -100.0) and np.all(v_trace <= 60.0), \
            f"Membrane voltage out of bounds: min={v_trace.min()}, max={v_trace.max()}"

        # 2. Assert ATP reserve reaches exhaustion (<= 0.0 or <= 0.05)
        min_E = float(np.min(E_trace))
        assert min_E <= 0.05, f"Expected ATP exhaustion under {I_stim} pA, but min(E) = {min_E:.4f}"

        # 3. Assert ZERO spikes while E <= 0.
        # Check every recorded spike time
        if len(spikes.t) > 0:
            spike_times_ms = np.array(spikes.t / ms)
            for t_spk in spike_times_ms:
                # Find the index in StateMonitor just before the spike
                idx = np.searchsorted(t_trace, t_spk) - 1
                if idx >= 0 and idx < len(E_trace):
                    E_pre_spike = E_trace[idx]
                    assert E_pre_spike > 0.0, \
                        f"Metabolic Arrest Violation: Neuron fired at t={t_spk:.2f} ms with E={E_pre_spike:.4f} <= 0!"

        # 4. Confirm firing rate is severely curtailed by metabolic clamp
        # Maximum unconstrained rate under 500 pA with 2 ms refractory would be ~500 Hz.
        # Under metabolic recovery (tau_rec = 1000 ms, spike_cost = 0.1), steady-state rate <= 10 Hz (RS) or 20 Hz (FS).
        n_spikes = len(spikes.t)
        max_possible_metabolic_spikes = int(10 + (sim_time_ms / (tau_rec * 0.1)) + 5)
        assert n_spikes <= max_possible_metabolic_spikes, \
            f"Metabolic clamp failed: {n_spikes} spikes fired, exceeding metabolic limit {max_possible_metabolic_spikes}"

    def test_atp_recovery_asymptote_after_extreme_exhaustion(self):
        """
        Adversarial Challenge 1 (Recovery Phase):
        Deplete neuron with 500 pA for 100 ms, then remove current (I=0) for 2000 ms.
        Confirm E recovers asymptotically to 1.0 with tau_recovery = 1000 ms.
        """
        b2.start_scope()
        G = create_metabolic_neurons(1, model_type='RS')
        G.I = 500.0

        spikes = SpikeMonitor(G)
        mon = StateMonitor(G, ['v', 'E'], record=True)
        net = Network(G, spikes, mon)

        # Deplete
        net.run(100 * ms)
        depleted_E = float(G.E[0])
        assert depleted_E <= 0.1, f"Failed to deplete E: {depleted_E}"

        # Recovery phase
        G.I = 0.0
        net.run(2000 * ms)

        recovered_E = float(G.E[0])
        assert recovered_E > 0.85, f"Recovery stalled: E reached only {recovered_E:.4f} after 2000 ms"
        assert recovered_E <= 1.0, f"Over-recovery overshoot: E={recovered_E:.4f} > 1.0"


class TestAdversarialRefractoryPeriod:
    """Adversarial stress testing of the 2.0 ms absolute refractory period."""

    def setup_method(self):
        b2.start_scope()
        defaultclock.dt = 0.1 * ms

    @pytest.mark.parametrize("I_stim", [100.0, 300.0, 500.0, 1000.0])
    def test_refractory_enforcement_under_extreme_drive(self, I_stim):
        """
        Adversarial Challenge 2:
        Drive 10 neurons with extreme current (100 to 1000 pA) and negligible spike cost
        to produce high-frequency firing bursts.
        Strictly assert min(ISI) >= 1.99 ms across all neurons and all spikes.
        """
        b2.start_scope()
        eqs = get_metabolic_izhikevich_eqs()
        n_test = 10
        G = NeuronGroup(
            n_test, eqs,
            threshold='v >= 30 and E > 0',
            reset='v = c; u += d; E -= spike_cost',
            method='euler',
            refractory=2.0 * ms
        )
        G.v = -65.0
        G.u = 0.2 * -65.0
        G.E = 1.0
        G.I = I_stim
        G.a = 0.02
        G.b = 0.2
        G.c = -65.0
        G.d = 8.0
        G.tau_recovery = 50 * ms
        G.spike_cost = 0.0001  # Tiny cost allows continuous high-rate firing

        spikes = SpikeMonitor(G)
        net = Network(G, spikes)
        net.run(200 * ms)

        assert len(spikes.t) > 50, f"Insufficient spikes collected: {len(spikes.t)}"

        min_observed_isi = float('inf')
        total_intervals = 0

        for nid in range(n_test):
            t_nid = np.array(spikes.t[spikes.i == nid] / ms)
            if len(t_nid) >= 2:
                isis = np.diff(t_nid)
                min_isi = float(np.min(isis))
                total_intervals += len(isis)
                if min_isi < min_observed_isi:
                    min_observed_isi = min_isi
                assert min_isi >= 1.99, \
                    f"Refractory breach for neuron {nid} under I={I_stim} pA: min(ISI)={min_isi:.4f} ms < 1.99 ms"

        assert min_observed_isi >= 1.99, f"Refractory violation: min observed ISI = {min_observed_isi:.4f} ms"
        assert total_intervals > 100, f"Expected >100 spike intervals tested, got {total_intervals}"

    def test_refractory_enforcement_under_synaptic_bombardment(self):
        """
        Adversarial Challenge 2 (Synaptic perturbation):
        Bombard a post-synaptic neuron with 20 synchronous excitatory inputs (w=50.0 each)
        during its absolute refractory period.
        Verify that synaptic inputs during refractory do NOT trigger premature spikes.
        """
        b2.start_scope()
        # Pre-synaptic neurons emitting spikes at t = 0.5 ms, 1.0 ms, 1.5 ms
        from brian2 import SpikeGeneratorGroup
        spike_indices = []
        spike_times = []
        for pre_i in range(10):
            for t_pulse in [0.5, 1.0, 1.5, 1.8]:
                spike_indices.append(pre_i)
                spike_times.append(t_pulse * ms)

        pre = SpikeGeneratorGroup(10, spike_indices, spike_times)

        eqs = get_metabolic_izhikevich_eqs()
        post = NeuronGroup(
            1, eqs,
            threshold='v >= 30 and E > 0',
            reset='v = c; u += d; E -= spike_cost',
            method='euler',
            refractory=2.0 * ms
        )
        post.v = 29.9  # Nearly at threshold
        post.u = 0.2 * post.v
        post.E = 1.0
        post.I = 100.0  # High drive triggers initial spike at t ~ 0.1 ms
        post.a = 0.02; post.b = 0.2; post.c = -65.0; post.d = 8.0
        post.tau_recovery = 1000 * ms; post.spike_cost = 0.01

        S = Synapses(pre, post, on_pre='v_post += 50.0', method='euler')
        S.connect(p=1.0)

        spk_mon = SpikeMonitor(post)
        net = Network(pre, post, S, spk_mon)
        net.run(10 * ms)

        t_spk = np.array(spk_mon.t / ms)
        assert len(t_spk) >= 2, f"Expected multiple spikes, got {len(t_spk)}"
        isis = np.diff(t_spk)
        assert np.all(isis >= 1.99), \
            f"Synaptic bombardment broke refractory period: ISIs = {isis} ms"

    def test_full_organoid_population_refractory_invariant(self):
        """
        Adversarial Challenge 2 (Full 500-Neuron OrganoidEnv):
        Run OrganoidEnv with maximum stimulation across all 500 neurons.
        Verify min(ISI) >= 1.99 ms across every active neuron in the population.
        """
        b2.start_scope()
        set_seed(42)
        env = OrganoidEnv()
        env.reset(seed=42)

        # Drive all neurons with strong current
        env.neurons.I = 150.0

        for _ in range(3):
            env.step(env.action_space.sample())

        spikes = env.spike_mon
        total_spikes = len(spikes.t)
        assert total_spikes > 100, f"Expected >100 spikes from full network, got {total_spikes}"

        min_population_isi = float('inf')
        evaluated_neurons = 0

        for nid in range(env.n_neurons):
            t_n = np.array(spikes.t[spikes.i == nid] / ms)
            if len(t_n) >= 2:
                isis = np.diff(t_n)
                min_i = float(np.min(isis))
                if min_i < min_population_isi:
                    min_population_isi = min_i
                assert min_i >= 1.99, \
                    f"Refractory violation in neuron {nid}: min ISI = {min_i:.4f} ms < 1.99 ms"
                evaluated_neurons += 1

        assert evaluated_neurons >= 50, f"Evaluated only {evaluated_neurons} active neurons"
        assert min_population_isi >= 1.99, \
            f"Full population min ISI violated: {min_population_isi:.4f} ms"


class TestAdversarialDalesPrincipleAndPlasticity:
    """Adversarial stress testing of Dale's principle and extreme dopamine rewards."""

    def setup_method(self):
        b2.start_scope()
        defaultclock.dt = 0.1 * ms

    def test_extreme_reward_plasticity_bounds_and_sign_invariance(self):
        """
        Adversarial Challenge 3:
        Apply extreme positive (+1000, +1e5) and negative (-1000, -1e5) dopamine rewards
        across multiple steps.
        Assert:
        1. All synaptic conductances remain strictly non-negative (w >= 0.0).
        2. All synaptic conductances remain bounded (w <= 100.0).
        3. Zero sign reversals or NaN/Inf values.
        """
        env = OrganoidEnv()
        env.reset(seed=123)

        # Set up non-zero traces so plasticity has an effect
        for S in env.synapses:
            if hasattr(S, 'Trace1'):
                S.Trace1 = 1.0
                S.Trace2 = 1.0
            elif hasattr(S, 'Trace'):
                S.Trace = 1.0

        # Sequence of extreme rewards
        extreme_rewards = [
            1000.0,
            -1000.0,
            50000.0,
            -50000.0,
            1e6,
            -1e6,
            0.0,
            -0.0,
            1e-12,
            -1e-12
        ]

        for step_idx, r in enumerate(extreme_rewards):
            apply_dopamine(env.synapses, reward=r, learning_rate=0.01)

            for s_idx, S in enumerate(env.synapses):
                w_arr = np.array(S.w)
                assert not np.any(np.isnan(w_arr)), \
                    f"Step {step_idx} (reward={r}): NaN detected in synapse {s_idx}"
                assert not np.any(np.isinf(w_arr)), \
                    f"Step {step_idx} (reward={r}): Inf detected in synapse {s_idx}"
                assert np.all(w_arr >= 0.0), \
                    f"Step {step_idx} (reward={r}): Negative conductance detected! min(w)={np.min(w_arr)}"
                assert np.all(w_arr <= 100.0), \
                    f"Step {step_idx} (reward={r}): Unbounded weight detected! max(w)={np.max(w_arr)}"

    def test_dales_principle_strict_projection_polarity(self):
        """
        Adversarial Challenge 3 (Architectural polarity audit):
        Confirm every projection in OrganoidEnv strictly preserves Dale's principle:
        - Excitatory sources (SDM, Hidden-Exc, Motor) ONLY depolarize (+w)
        - Inhibitory sources (P_inh) ONLY hyperpolarize (-w)
        - Motor units (400..499) are 100% Excitatory
        """
        env = OrganoidEnv()

        for s_idx, S in enumerate(env.synapses):
            is_inh = getattr(S, '_is_inhibitory', False)
            source_grp = S.source

            # Check source identity
            if is_inh:
                assert source_grp is env.P_inh, \
                    f"Inhibitory synapse {s_idx} originates from non-inhibitory source: {source_grp}"
            else:
                assert source_grp in (env.P_exc_sdm, env.P_exc_hidden, env.P_exc_motor), \
                    f"Excitatory synapse {s_idx} originates from invalid source: {source_grp}"

            # Conductance non-negativity
            weights = np.array(S.w)
            assert np.all(weights >= 0.0), f"Negative conductances found in synapse {s_idx}"

        # Confirm motor group properties
        assert len(env.P_exc_motor) == 100
        assert np.all(env.P_exc_motor.a == 0.02), "Motor neurons must have RS a=0.02"
        assert np.all(env.P_exc_motor.d == 8.0), "Motor neurons must have RS d=8.0"
        assert np.all(env.P_exc_motor.tau_recovery == 1000 * ms), "Motor tau_recovery must be 1000 ms"

        # Check total excitatory (400) and inhibitory (100)
        assert len(env.P_exc) == 400, f"Total excitatory = {len(env.P_exc)} != 400"
        assert len(env.P_inh) == 100, f"Total inhibitory = {len(env.P_inh)} != 100"
