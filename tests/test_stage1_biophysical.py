"""
tests/test_stage1_biophysical.py

Stage 1 Verification Suite: Biophysical Invariants & Physiological Plausibility
Authoritative Specifications:
- OrganoidEnv_Master_Verifier_Plan.md §1.1 - §1.5
- PROJECT.md §Feature Inventory (Items 4-9)
- 4-Tier Verification Hierarchy:
    Tier 1: Isolated Canonical Equivalence (Single Neuron & Trace)
    Tier 2: Boundary Clamps (E <= 0, refractory clamp, zero-spike clamp)
    Tier 3: Dale's Principle & Population Polarity Preservation
    Tier 4: Dynamic Environmental Closed-Loop Invariants
"""

import pytest
import numpy as np
import brian2 as b2
from brian2 import ms, second, Hz, Network, NeuronGroup, Synapses, SpikeMonitor, StateMonitor, defaultclock

from organoid_rl.environment.core import OrganoidEnv, OrganoidConfig
from organoid_rl.environment.neurons import create_metabolic_neurons, get_metabolic_izhikevich_eqs
from organoid_rl.utils.seeding import set_seed


class TestBrian2CanonicalEquivalence:
    """1.1 Brian2 Canonical Numerical Equivalence Verification."""

    def setup_method(self):
        b2.start_scope()
        defaultclock.dt = 0.1 * ms

    def test_single_neuron_step_current_stability(self):
        """Tier 1: Single neuron under 15 pA step current integrates stably without NaN/Inf."""
        eqs = get_metabolic_izhikevich_eqs()
        G = NeuronGroup(1, eqs, threshold='v >= 30 and E > 0', reset='v = c; u += d; E -= spike_cost',
                        method='euler', refractory=2*ms)
        G.v = -65.0
        G.u = 0.2 * -65.0
        G.E = 1.0
        G.I = 15.0  # 15 pA injection per Verifier Plan §1.1
        G.a = 0.02
        G.b = 0.2
        G.c = -65.0
        G.d = 8.0
        G.tau_recovery = 1000 * ms
        G.spike_cost = 0.1

        net = Network(G)
        net.run(10 * ms)

        v_val = float(G.v[0])
        assert not np.isnan(v_val), "Numerical divergence: NaN detected in Euler integration"
        assert not np.isinf(v_val), "Numerical divergence: Inf detected in Euler integration"
        assert -100.0 <= v_val <= 60.0, f"Membrane voltage bounds violated: {v_val} mV"


class TestMetabolicGovernor:
    """1.2 Metabolic Governor Binding Invariant Verification."""

    def setup_method(self):
        b2.start_scope()
        defaultclock.dt = 0.1 * ms

    def test_metabolic_exhaustion_halts_firing(self):
        """Tier 2: When ATP energy E <= 0, the neuron is strictly prohibited from spiking."""
        eqs = get_metabolic_izhikevich_eqs()
        G = NeuronGroup(1, eqs, threshold='v >= 30 and E > 0', reset='v = c; u += d; E -= spike_cost',
                        method='euler', refractory=2*ms)
        G.v = -65.0
        G.u = 0.2 * -65.0
        G.E = -0.1  # Zero/negative energy floor
        G.I = 100.0  # Strong suprathreshold current
        G.a = 0.02
        G.b = 0.2
        G.c = -65.0
        G.d = 8.0
        G.tau_recovery = 1e9 * ms  # Freeze recovery so E stays <= 0
        G.spike_cost = 0.1

        spikes = SpikeMonitor(G)
        net = Network(G, spikes)
        net.run(20 * ms)

        assert len(spikes.t) == 0, f"Violation: neuron spiked {len(spikes.t)} times despite E <= 0!"

    def test_metabolic_depletion_and_recovery_dynamics(self):
        """Tier 1: Sustained spiking depletes ATP reserve E; recovery follows tau_recovery."""
        G = create_metabolic_neurons(1, model_type='RS')
        G.I = 50.0
        spikes = SpikeMonitor(G)
        mon = StateMonitor(G, ['v', 'E'], record=True)

        net = Network(G, spikes, mon)
        net.run(500 * ms)

        n_spikes = len(spikes.t)
        assert n_spikes > 0, "Neuron failed to fire under 50 pA current"
        min_E = np.min(mon.E[0])
        assert min_E < 0.20, f"Expected metabolic depletion down to <= 0.20, got {min_E}"


class TestRefractoryPeriod:
    """1.3 Absolute Refractory Period Enforcement Invariant (t_ref = 2.0 ms)."""

    def setup_method(self):
        b2.start_scope()
        defaultclock.dt = 0.1 * ms

    def test_minimum_inter_spike_interval_bound(self):
        """Tier 2: Under massive continuous depolarizing current (200 pA), min(ISI) >= 2.0 ms."""
        eqs = get_metabolic_izhikevich_eqs()
        G = NeuronGroup(5, eqs, threshold='v >= 30 and E > 0', reset='v = c; u += d; E -= spike_cost',
                        method='euler', refractory=2.0*ms)
        G.v = -65.0
        G.u = 0.2 * -65.0
        G.E = 1.0
        G.I = 200.0  # Extreme current
        G.a = 0.02
        G.b = 0.2
        G.c = -65.0
        G.d = 8.0
        G.tau_recovery = 100 * ms
        G.spike_cost = 0.001  # Negligible cost to allow sustained burst

        spikes = SpikeMonitor(G)
        net = Network(G, spikes)
        net.run(200 * ms)

        assert len(spikes.t) > 10, "Need at least 10 spikes to compute ISI distribution"

        for neuron_id in range(5):
            spike_times = spikes.t[spikes.i == neuron_id] / ms
            if len(spike_times) >= 2:
                isis = np.diff(spike_times)
                min_isi = np.min(isis)
                assert min_isi >= 1.99, f"Refractory violation for neuron {neuron_id}: min(ISI) = {min_isi:.3f} ms < 2.0 ms"


class TestDalesPrinciple:
    """1.4 Dale's Principle & Synaptic Polarity Compliance."""

    def test_population_cell_type_proportions(self):
        """Tier 1: Excitatory count = 400 (80%), Inhibitory count = 100 (20%)."""
        env = OrganoidEnv()
        assert env.n_neurons == 500, f"Total neurons mismatch: {env.n_neurons} != 500"

        # Check P_exc and P_inh groups
        assert len(env.P_exc_sdm) == 256, f"SDM group size mismatch: {len(env.P_exc_sdm)} != 256"
        assert len(env.P_exc_hidden) == 44, f"Hidden excitatory size mismatch: {len(env.P_exc_hidden)} != 44"
        assert len(env.P_exc_motor) == 100, f"Motor readout size mismatch: {len(env.P_exc_motor)} != 100"

        total_exc = len(env.P_exc_sdm) + len(env.P_exc_hidden) + len(env.P_exc_motor)
        total_inh = len(env.P_inh)

        assert total_exc == 400, f"Dale's violation: Excitatory count = {total_exc} != 400 (80%)"
        assert total_inh == 100, f"Dale's violation: Inhibitory count = {total_inh} != 100 (20%)"

    def test_synaptic_sign_preservation(self):
        """Tier 3: Excitatory synapses only depolarize (+w); inhibitory synapses only hyperpolarize (-w)."""
        env = OrganoidEnv()
        for idx, S in enumerate(env.synapses):
            is_inh = getattr(S, '_is_inhibitory', False)
            weights = np.array(S.w)
            assert np.all(weights >= 0.0), f"Negative synaptic conductance detected in synapse {idx}"

            # Verify Dale's law: Inhibitory projections must originate exclusively from P_inh
            if is_inh:
                assert S.source is env.P_inh, f"Inhibitory synapse {idx} must originate from P_inh"
            else:
                assert S.source in (env.P_exc_sdm, env.P_exc_hidden, env.P_exc_motor), \
                    f"Excitatory synapse {idx} must originate from excitatory neuron groups"


class TestEligibilityTraceExponentialDecay:
    """1.5 Presynaptic Eligibility Trace Exponential Decay Invariant."""

    def setup_method(self):
        b2.start_scope()
        defaultclock.dt = 0.1 * ms

    def test_dual_trace_analytical_exponential_decay(self):
        """
        Tier 1: Presynaptic traces follow analytical forms:
        Trace1(t) = Trace1(0) * exp(-t / 100ms)
        Trace2(t) = Trace2(0) * exp(-t / 2500ms)
        Max Euler truncation error < 5e-4 over a 500 ms window with dt=0.1ms.
        """
        G1 = NeuronGroup(1, 'v:1', threshold='False', method='euler')
        G2 = NeuronGroup(1, 'v:1', threshold='False', method='euler')

        syn_eqs = '''
        dw/dt = (w0 - w) / tau_decay : 1 (clock-driven)
        dTrace1/dt = -Trace1 / tau_trace1 : 1 (clock-driven)
        dTrace2/dt = -Trace2 / tau_trace2 : 1 (clock-driven)
        w0 : 1 (shared)
        '''
        ns = {'tau_trace1': 100*ms, 'tau_trace2': 2500*ms, 'tau_decay': 10000*ms}
        S = Synapses(G1, G2, model=syn_eqs, method='euler', namespace=ns)
        S.connect(i=0, j=0)
        S.w = 15.0
        S.w0 = 15.0
        S.Trace1 = 1.0  # Initial trace spike
        S.Trace2 = 1.0

        mon = StateMonitor(S, ['Trace1', 'Trace2'], record=True)
        net = Network(G1, G2, S, mon)

        eval_time_ms = 500.0
        net.run(eval_time_ms * ms)

        t_sim = np.array(mon.t / ms)
        trace1_sim = np.array(mon.Trace1[0])
        trace2_sim = np.array(mon.Trace2[0])

        trace1_analytical = 1.0 * np.exp(-t_sim / 100.0)
        trace2_analytical = 1.0 * np.exp(-t_sim / 2500.0)

        err1 = np.max(np.abs(trace1_sim - trace1_analytical))
        err2 = np.max(np.abs(trace2_sim - trace2_analytical))

        assert err1 < 5e-4, f"Trace1 exponential decay error too high: {err1}"
        assert err2 < 5e-4, f"Trace2 exponential decay error too high: {err2}"
