/*
 * _snn_kernel.c
 * High-performance C integration kernel for OrganoidEnv.
 *
 * Matches Brian2's exact execution ordering, Euler Izhikevich+K_ATP dynamics,
 * exact/Euler synaptic decay, CSR presynaptic transmission, refractory checks,
 * and Bernoulli/Poisson background drive.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

static inline double clip_val(double x, double low, double high) {
    if (x < low) return low;
    if (x > high) return high;
    return x;
}

static inline int64_t brian_timestep(double t, double dt) {
    return (int64_t)((t + 1e-3 * dt) / dt);
}

static inline uint64_t xorshift64star(uint64_t *state) {
    uint64_t x = *state;
    if (x == 0) x = 0x9E3779B97F4A7C15ULL;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

EXPORT int32_t run_snn_ticks(
    int32_t n_ticks,
    int32_t n_neurons,
    double dt,
    int64_t *timestep_ptr,
    double *t_ptr,
    double refractory_sec,
    double *v,
    double *u,
    double *E,
    const double *I_ext,
    const double *a,
    const double *b,
    const double *c,
    const double *d,
    const double *tau_recovery,
    const double *spike_cost,
    double *lastspike,
    uint8_t *not_refractory,
    int32_t *spikespace,
    int32_t n_groups,
    const int32_t *g_M,
    const int32_t *g_sign,
    const double *g_w0,
    double **g_w,
    double **g_t1,
    double **g_t2,
    int32_t **g_indptr,
    int32_t **g_post,
    int32_t **g_syn_idx,
    int32_t use_dual_trace,
    int32_t frozen_weights,
    double tau_trace1,
    double tau_trace2,
    double tau_trace_single,
    double tau_decay,
    double poisson_rate_hz,
    double poisson_weight,
    uint64_t *rng_state,
    double *out_spike_t,
    int32_t *out_spike_i,
    int32_t max_out_spikes,
    int32_t neuron_mode,
    double v_thresh,
    double tau_m_sec
) {
    if (n_ticks <= 0) {
        return 0;
    }

    const int64_t lio_ref_ticks = brian_timestep(refractory_sec, dt);
    const double lio_2 = dt / 0.001; /* dt / ms */
    const double lif_dt_ratio = dt / tau_m_sec;

    const double factor_t1 = use_dual_trace ? exp(-dt / tau_trace1) : (1.0 - dt / tau_trace_single);
    const double factor_t2 = use_dual_trace ? exp(-dt / tau_trace2) : 1.0;
    const double factor_decay = use_dual_trace ? exp(-dt / tau_decay) : (1.0 - dt / tau_decay);

    double *pow_t1 = (double *)malloc((size_t)(n_ticks + 1) * sizeof(double));
    double *pow_t2 = (double *)malloc((size_t)(n_ticks + 1) * sizeof(double));
    double *pow_decay = (double *)malloc((size_t)(n_ticks + 1) * sizeof(double));
    pow_t1[0] = 1.0;
    pow_t2[0] = 1.0;
    pow_decay[0] = 1.0;
    for (int32_t k = 1; k <= n_ticks; k++) {
        pow_t1[k] = pow_t1[k - 1] * factor_t1;
        pow_t2[k] = pow_t2[k - 1] * factor_t2;
        pow_decay[k] = pow_decay[k - 1] * factor_decay;
    }

    /* Pre-decay all eligibility traces once across full n_ticks chunk */
    {
        const double pt1 = pow_t1[n_ticks];
        const double pt2 = pow_t2[n_ticks];
        for (int32_t g = 0; g < n_groups; g++) {
            const int32_t M = g_M[g];
            double * restrict t1 = g_t1[g];
            for (int32_t s = 0; s < M; s++) {
                t1[s] *= pt1;
            }
            if (use_dual_trace) {
                double * restrict t2 = g_t2[g];
                for (int32_t s = 0; s < M; s++) {
                    t2[s] *= pt2;
                }
            }
        }
    }

    int32_t *g_offsets = (int32_t *)malloc((size_t)(n_groups + 1) * sizeof(int32_t));
    g_offsets[0] = 0;
    for (int32_t g = 0; g < n_groups; g++) {
        g_offsets[g + 1] = g_offsets[g] + g_M[g];
    }
    const int32_t total_syn = g_offsets[n_groups];
    int32_t *last_tick = (!frozen_weights && total_syn > 0)
        ? (int32_t *)calloc((size_t)total_syn, sizeof(int32_t))
        : NULL;

    const double poisson_p = poisson_rate_hz * dt;
    /* Threshold mapped onto 64-bit uniform [0, 2^64 - 1] */
    const uint64_t poisson_thresh = (poisson_p >= 1.0)
        ? UINT64_MAX
        : (poisson_p <= 0.0 ? 0ULL : (uint64_t)(poisson_p * 18446744073709551616.0));

    int32_t total_out_spikes = 0;

    for (int32_t step = 0; step < n_ticks; step++) {
        const double t_now = (*timestep_ptr) * dt;
        *t_ptr = t_now;

        /* 1. NeuronGroup stateupdater (groups, order=0) */
        if (neuron_mode == 1) {
            /* Canonical LIF dynamics */
            for (int32_t i = 0; i < n_neurons; i++) {
                uint8_t nr = (brian_timestep(t_now - lastspike[i], dt) >= lio_ref_ticks) ? 1 : 0;
                if (nr) {
                    v[i] = v[i] + lif_dt_ratio * (-(v[i] + 65.0) + I_ext[i]);
                }
                not_refractory[i] = nr;
            }
        } else {
            /* Metabolic-Izhikevich dynamics */
            for (int32_t i = 0; i < n_neurons; i++) {
                uint8_t nr = (brian_timestep(t_now - lastspike[i], dt) >= lio_ref_ticks) ? 1 : 0;
                double e_val = E[i];
                double v_val = v[i];
                double u_val = u[i];
                double vc = clip_val(v_val, -100.0, 60.0);

                double new_E = e_val + (dt * (1.0 - e_val)) / tau_recovery[i];
                double new_u = lio_2 * (a[i] * (b[i] * vc - u_val)) + u_val;

                if (nr) {
                    double new_v;
                    if (e_val <= 0.0) {
                        new_v = v_val + lio_2 * (-160.0 - 2.0 * vc);
                    } else {
                        new_v = v_val + lio_2 * (140.0 + I_ext[i] + 0.04 * vc * vc + 5.0 * vc - u_val);
                    }
                    v[i] = new_v;
                }
                E[i] = new_E;
                u[i] = new_u;
                not_refractory[i] = nr;
            }
        }

        /* 2. Synaptic decays are algebraically batched via lazy catch-up & end-of-call pass */

        /* 3. Thresholder (thresholds, order=0) */
        int32_t num_events = 0;
        for (int32_t i = 0; i < n_neurons; i++) {
            if (not_refractory[i] && v[i] >= v_thresh && E[i] > 0.0) {
                spikespace[num_events++] = i;
                not_refractory[i] = 0;
                lastspike[i] = t_now;
            }
        }
        spikespace[n_neurons] = num_events;

        /* 4. SpikeMonitor append (thresholds, order=1) */
        for (int32_t k = 0; k < num_events; k++) {
            if (total_out_spikes < max_out_spikes) {
                out_spike_t[total_out_spikes] = t_now;
                out_spike_i[total_out_spikes] = spikespace[k];
                total_out_spikes++;
            }
        }

        /* 5. SynapticPathway on_pre (synapses, order=-1) */
        if (num_events > 0) {
            const double add_t1 = pow_t1[n_ticks - 1 - step];
            const double add_t2 = pow_t2[n_ticks - 1 - step];
            const int32_t tick_next = step + 1;
            for (int32_t g = 0; g < n_groups; g++) {
                const double sign = (double)g_sign[g];
                const double w0_val = g_w0[g];
                const int32_t *indptr = g_indptr[g];
                const int32_t *csr_post = g_post[g];
                const int32_t *csr_syn = g_syn_idx[g];
                double *w = g_w[g];
                double *t1 = g_t1[g];
                double *t2 = g_t2[g];
                int32_t *lt = last_tick ? (last_tick + g_offsets[g]) : NULL;

                for (int32_t k = 0; k < num_events; k++) {
                    int32_t pre_id = spikespace[k];
                    int32_t e_start = indptr[pre_id];
                    int32_t e_end = indptr[pre_id + 1];
                    for (int32_t e = e_start; e < e_end; e++) {
                        int32_t post_id = csr_post[e];
                        int32_t s_idx = csr_syn[e];
                        if (lt) {
                            int32_t delta = tick_next - lt[s_idx];
                            if (delta > 0) {
                                w[s_idx] = w0_val + (w[s_idx] - w0_val) * pow_decay[delta];
                                lt[s_idx] = tick_next;
                            }
                        }
                        if (not_refractory[post_id]) {
                            v[post_id] += sign * w[s_idx];
                        }
                        t1[s_idx] += add_t1;
                        if (use_dual_trace) {
                            t2[s_idx] += add_t2;
                        }
                    }
                }
            }
        }

        /* 6. PoissonInput background drive (synapses, order=0) */
        if (poisson_thresh > 0ULL) {
            for (int32_t i = 0; i < n_neurons; i++) {
                uint64_t r = xorshift64star(rng_state);
                if (not_refractory[i] && r < poisson_thresh) {
                    v[i] += poisson_weight;
                }
            }
        }

        /* 7. Spike resetter (resets, order=0) */
        for (int32_t k = 0; k < num_events; k++) {
            int32_t idx = spikespace[k];
            v[idx] = c[idx];
            if (neuron_mode == 0) {
                u[idx] += d[idx];
                E[idx] -= spike_cost[idx];
            }
        }

        /* 8. Advance clock */
        (*timestep_ptr) += 1;
        *t_ptr = (*timestep_ptr) * dt;
    }

    /* Finalize remaining exponential weight decay at end of chunk */
    if (last_tick) {
        for (int32_t g = 0; g < n_groups; g++) {
            const int32_t M = g_M[g];
            const double w0_val = g_w0[g];
            double * restrict w = g_w[g];
            int32_t * restrict lt = last_tick + g_offsets[g];
            for (int32_t s = 0; s < M; s++) {
                int32_t rem = n_ticks - lt[s];
                if (rem > 0) {
                    w[s] = w0_val + (w[s] - w0_val) * pow_decay[rem];
                }
            }
        }
        free(last_tick);
    }
    free(g_offsets);
    free(pow_t1);
    free(pow_t2);
    free(pow_decay);

    return total_out_spikes;
}
