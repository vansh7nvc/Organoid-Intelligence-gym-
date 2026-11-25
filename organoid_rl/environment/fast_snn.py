"""
Native C-accelerated SNN integration backend for OrganoidEnv.

Compiles `organoid_rl/environment/_snn_kernel.c` once using GCC (`-O3`) when
available and executes Brian2's exact coupled ODE + CSR spike transmission
directly on Brian2's live NumPy state arrays with zero Python neuron/synapse
loops. Falls back seamlessly to Brian2 `network.run` if GCC is unavailable.
"""

import ctypes
import os
import platform
import shutil
import subprocess
import numpy as np
from brian2 import ms, second


_KERNEL_LIB = None
_KERNEL_FUNC = None
_LOAD_ATTEMPTED = False


def _splitmix64(x: int) -> int:
    """Deterministic 64-bit state mixer for seeding xorshift64*."""
    z = (int(x) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF
    return z if z != 0 else 0x9E3779B97F4A7C15


def _ensure_gcc_on_path():
    if platform.system() == "Windows":
        msys_bin = r"C:\msys64\ucrt64\bin"
        if os.path.isdir(msys_bin) and msys_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = msys_bin + os.pathsep + os.environ.get("PATH", "")


def load_snn_kernel():
    """Compiles (if needed) and loads `_snn_kernel` shared library."""
    global _KERNEL_LIB, _KERNEL_FUNC, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _KERNEL_FUNC
    _LOAD_ATTEMPTED = True

    _ensure_gcc_on_path()
    gcc_path = shutil.which("gcc")
    env_dir = os.path.dirname(os.path.abspath(__file__))
    c_path = os.path.join(env_dir, "_snn_kernel.c")
    ext = ".dll" if platform.system() == "Windows" else ".so"
    lib_path = os.path.join(env_dir, f"_snn_kernel{ext}")

    if not os.path.exists(c_path):
        return None

    need_compile = not os.path.exists(lib_path) or (
        os.path.getmtime(c_path) > os.path.getmtime(lib_path)
    )
    if need_compile:
        if gcc_path is None:
            return None
        tmp_lib = lib_path + f".tmp_{os.getpid()}"
        cmd = [gcc_path, "-O3", "-march=native" if platform.machine().endswith("64") else "-O3", "-shared"]
        if platform.system() != "Windows":
            cmd.append("-fPIC")
        cmd.extend([c_path, "-o", tmp_lib, "-lm"])
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            os.replace(tmp_lib, lib_path)
        except Exception as exc:
            if os.path.exists(tmp_lib):
                try:
                    os.remove(tmp_lib)
                except OSError:
                    pass
            raise RuntimeError(f"Failed to compile {c_path}: {exc}") from exc

    try:
        lib = ctypes.CDLL(lib_path)
        fn = lib.run_snn_ticks
        fn.restype = ctypes.c_int32
        fn.argtypes = [
            ctypes.c_int32,                    # n_ticks
            ctypes.c_int32,                    # n_neurons
            ctypes.c_double,                   # dt
            ctypes.POINTER(ctypes.c_int64),    # timestep_ptr
            ctypes.POINTER(ctypes.c_double),   # t_ptr
            ctypes.c_double,                   # refractory_sec
            ctypes.POINTER(ctypes.c_double),   # v
            ctypes.POINTER(ctypes.c_double),   # u
            ctypes.POINTER(ctypes.c_double),   # E
            ctypes.POINTER(ctypes.c_double),   # I_ext
            ctypes.POINTER(ctypes.c_double),   # a
            ctypes.POINTER(ctypes.c_double),   # b
            ctypes.POINTER(ctypes.c_double),   # c
            ctypes.POINTER(ctypes.c_double),   # d
            ctypes.POINTER(ctypes.c_double),   # tau_recovery
            ctypes.POINTER(ctypes.c_double),   # spike_cost
            ctypes.POINTER(ctypes.c_double),   # lastspike
            ctypes.POINTER(ctypes.c_uint8),    # not_refractory
            ctypes.POINTER(ctypes.c_int32),    # spikespace
            ctypes.c_int32,                    # n_groups
            ctypes.POINTER(ctypes.c_int32),    # g_M
            ctypes.POINTER(ctypes.c_int32),    # g_sign
            ctypes.POINTER(ctypes.c_double),   # g_w0
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # g_w
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # g_t1
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # g_t2
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),   # g_indptr
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),   # g_post
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),   # g_syn_idx
            ctypes.c_int32,                    # use_dual_trace
            ctypes.c_int32,                    # frozen_weights
            ctypes.c_double,                   # tau_trace1
            ctypes.c_double,                   # tau_trace2
            ctypes.c_double,                   # tau_trace_single
            ctypes.c_double,                   # tau_decay
            ctypes.c_double,                   # poisson_rate_hz
            ctypes.c_double,                   # poisson_weight
            ctypes.POINTER(ctypes.c_uint64),   # rng_state
            ctypes.POINTER(ctypes.c_double),   # out_spike_t
            ctypes.POINTER(ctypes.c_int32),    # out_spike_i
            ctypes.c_int32,                    # max_out_spikes
            ctypes.c_int32,                    # neuron_mode
            ctypes.c_double,                   # v_thresh
            ctypes.c_double,                   # tau_m_sec
        ]
        _KERNEL_LIB = lib
        _KERNEL_FUNC = fn
        return fn
    except Exception:
        return None


class FastSNNRunner:
    """
    Executes Brian2-equivalent time integration directly against OrganoidEnv's
    Brian2 NeuronGroup, Synapses, and SpikeMonitor backing arrays.
    """

    def __init__(self, env, kernel_fn):
        self.env = env
        self.kernel_fn = kernel_fn
        self.n_neurons = int(env.n_neurons)
        self.n_groups = len(env.synapses)
        self.use_dual_trace = 1 if env.use_dual_trace else 0
        self.frozen_weights = 1 if (not env.use_plasticity or env.random_snn) else 0
        self.neuron_mode = 1 if getattr(env, "neuron_model", "metabolic_izhikevich") == "lif" else 0
        self.v_thresh = -50.0 if self.neuron_mode == 1 else 30.0
        self.tau_m_sec = 0.02
        self.tau_trace_single = float(getattr(env, "tau_trace", 50.0 * ms) / second)

        self.g_M = np.zeros(self.n_groups, dtype=np.int32)
        self.g_sign = np.zeros(self.n_groups, dtype=np.int32)
        self.g_w0 = np.zeros(self.n_groups, dtype=np.float64)

        self._indptrs = []
        self._posts = []
        self._syn_idxs = []

        for g, S in enumerate(env.synapses):
            pre_offset = int(getattr(S.source, "start", 0))
            post_offset = int(getattr(S.target, "start", 0))
            i_local = np.asarray(S.i[:], dtype=np.int32)
            j_local = np.asarray(S.j[:], dtype=np.int32)
            pre_global = i_local + pre_offset
            post_global = j_local + post_offset

            order = np.argsort(pre_global, kind="stable")
            counts = np.bincount(pre_global, minlength=self.n_neurons)
            indptr = np.zeros(self.n_neurons + 1, dtype=np.int32)
            np.cumsum(counts, out=indptr[1:])

            csr_post = np.ascontiguousarray(post_global[order], dtype=np.int32)
            csr_syn_idx = np.ascontiguousarray(order, dtype=np.int32)

            self.g_M[g] = len(i_local)
            self.g_sign[g] = -1 if getattr(S, "_is_inhibitory", False) else 1
            w0_arr = np.atleast_1d(np.asarray(S.variables["w0"].get_value(), dtype=np.float64))
            self.g_w0[g] = float(w0_arr[0]) if len(w0_arr) > 0 else 0.0

            self._indptrs.append(indptr)
            self._posts.append(csr_post)
            self._syn_idxs.append(csr_syn_idx)

        self._ptr_indptr = (ctypes.POINTER(ctypes.c_int32) * self.n_groups)(
            *[arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)) for arr in self._indptrs]
        )
        self._ptr_post = (ctypes.POINTER(ctypes.c_int32) * self.n_groups)(
            *[arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)) for arr in self._posts]
        )
        self._ptr_syn_idx = (ctypes.POINTER(ctypes.c_int32) * self.n_groups)(
            *[arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)) for arr in self._syn_idxs]
        )

        # Pre-allocate spike output scratch buffers (up to 500k spikes per step call)
        self._max_out_spikes = 500000
        self._out_spike_t = np.empty(self._max_out_spikes, dtype=np.float64)
        self._out_spike_i = np.empty(self._max_out_spikes, dtype=np.int32)
        self._dummy_t2 = np.zeros(1, dtype=np.float64)

        self._rng_state = np.array([_splitmix64(42)], dtype=np.uint64)

    def seed(self, seed_val: int):
        self._rng_state[0] = _splitmix64(int(seed_val))

    def get_rng_state(self) -> int:
        return int(self._rng_state[0])

    def set_rng_state(self, state_val: int):
        val = int(state_val)
        self._rng_state[0] = val if val != 0 else 0x9E3779B97F4A7C15

    def run_duration(self, duration_quantity, poisson_rate_hz: float = 10.0, poisson_weight: float = 4.0):
        """
        Integrates `duration_quantity` directly on Brian2's underlying state arrays
        and appends recorded spikes to `self.env.spike_mon`.
        """
        neurons = self.env.neurons
        net = self.env.network
        clock = neurons.clock

        dt_sec = float(clock.dt / second)
        dur_sec = float(duration_quantity / second)
        n_ticks = int(round(dur_sec / dt_sec))
        if n_ticks <= 0:
            return

        req_cap = max(500000, n_ticks * self.n_neurons + 1024)
        if req_cap > self._max_out_spikes:
            self._max_out_spikes = req_cap
            self._out_spike_t = np.empty(self._max_out_spikes, dtype=np.float64)
            self._out_spike_i = np.empty(self._max_out_spikes, dtype=np.int32)

        # Note: Do NOT trigger before_run; read live backing NumPy arrays directly.
        timestep_arr = clock.variables["timestep"].get_value()
        t_arr = clock.variables["t"].get_value()

        v_arr = neurons.variables["v"].get_value()
        u_arr = neurons.variables["u"].get_value()
        E_arr = neurons.variables["E"].get_value()
        I_arr = neurons.variables["I"].get_value()
        a_arr = neurons.variables["a"].get_value()
        b_arr = neurons.variables["b"].get_value()
        c_arr = neurons.variables["c"].get_value()
        d_arr = neurons.variables["d"].get_value()
        tau_rec_arr = neurons.variables["tau_recovery"].get_value()
        spk_cost_arr = neurons.variables["spike_cost"].get_value()
        lastspike_arr = neurons.variables["lastspike"].get_value()
        not_ref_arr = neurons.variables["not_refractory"].get_value()
        spikespace_arr = neurons.variables["_spikespace"].get_value()

        w_ptrs = (ctypes.POINTER(ctypes.c_double) * self.n_groups)()
        t1_ptrs = (ctypes.POINTER(ctypes.c_double) * self.n_groups)()
        t2_ptrs = (ctypes.POINTER(ctypes.c_double) * self.n_groups)()

        for g, S in enumerate(self.env.synapses):
            w_arr = S.variables["w"].get_value()
            w_ptrs[g] = w_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            if self.use_dual_trace:
                t1_arr = S.variables["Trace1"].get_value()
                t2_arr = S.variables["Trace2"].get_value()
            else:
                t1_arr = S.variables["Trace"].get_value()
                t2_arr = self._dummy_t2
            t1_ptrs[g] = t1_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            t2_ptrs[g] = t2_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        refractory_sec = float(self.env.refractory_period) * 0.001

        n_new = self.kernel_fn(
            n_ticks,
            self.n_neurons,
            dt_sec,
            timestep_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            t_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            refractory_sec,
            v_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            u_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            E_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            I_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            a_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            b_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            c_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            d_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            tau_rec_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            spk_cost_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            lastspike_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            not_ref_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            spikespace_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self.n_groups,
            self.g_M.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self.g_sign.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self.g_w0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            w_ptrs,
            t1_ptrs,
            t2_ptrs,
            self._ptr_indptr,
            self._ptr_post,
            self._ptr_syn_idx,
            self.use_dual_trace,
            self.frozen_weights,
            0.1,    # tau_trace1 = 100 ms
            2.5,    # tau_trace2 = 2500 ms
            self.tau_trace_single,
            10.0,   # tau_decay = 10000 ms
            poisson_rate_hz,
            poisson_weight,
            self._rng_state.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            self._out_spike_t.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._out_spike_i.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self._max_out_spikes,
            self.neuron_mode,
            self.v_thresh,
            self.tau_m_sec,
        )

        # Synchronize network clock time
        net.t_ = float(t_arr[0])

        # Append generated spikes cleanly into Brian2 SpikeMonitor backing DynamicArrays
        if n_new > 0 and self.env.spike_mon is not None:
            sm = self.env.spike_mon
            t_var = sm.variables["t"]
            i_var = sm.variables["i"]
            N_var = sm.variables["N"]
            count_var = sm.variables["count"]

            n_prev = int(N_var.get_value()[0])
            n_total = n_prev + int(n_new)
            t_var.resize(n_total)
            i_var.resize(n_total)

            t_buf = t_var.get_value()
            i_buf = i_var.get_value()
            t_buf[n_prev:n_total] = self._out_spike_t[:n_new]
            i_buf[n_prev:n_total] = self._out_spike_i[:n_new]
            N_var.get_value()[0] = n_total

            new_counts = np.bincount(self._out_spike_i[:n_new], minlength=self.n_neurons)
            count_var.get_value()[:] += new_counts
