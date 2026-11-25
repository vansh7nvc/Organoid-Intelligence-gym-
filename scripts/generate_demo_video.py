"""
Generates a real closed-loop OrganoidEnv demo animation (GIF + MP4) from a live
C99-accelerated Stage-4 episode:
  - 3D Rotating 500-Neuron Organoid Connectome (live spike flashes)
  - 2D Navigation Arena (obstacles, goal sphere, cursor trajectory)
  - Scrolling 500-Neuron Spike Raster
  - Population Firing Rate (Hz) & Energy Reserve E(t)

Outputs:
  - docs/assets/organoidenv_demo.gif
  - docs/assets/organoidenv_demo.mp4
"""

import os
import sys
import numpy as np
import brian2 as b2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import cv2

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from organoid_rl.environment.core import OrganoidEnv
from organoid_rl.platform.server import ACTION_VECTORS


def _seeded_rand(i: int, salt: int) -> float:
    x = np.sin(i * 12.9898 + salt * 78.233) * 43758.5453
    return float(x - np.floor(x))


def _build_3d_coords(n_neurons: int = 500):
    coords = np.zeros((n_neurons, 3), dtype=np.float64)
    base_colors = np.zeros((n_neurons, 3), dtype=np.float64)
    centers = [(0, 1.35, 0), (0, -1.35, 0), (-1.35, 0, 0), (1.35, 0, 0)]
    for i in range(n_neurons):
        u = _seeded_rand(i, 1) * np.pi * 2.0
        v = np.arccos(2.0 * _seeded_rand(i, 2) - 1.0)
        if i < 256:
            r = 1.65 + _seeded_rand(i, 3) * 0.35
            coords[i] = [r * np.sin(v) * np.cos(u), r * np.cos(v), r * np.sin(v) * np.sin(u)]
            base_colors[i] = [0.23, 0.51, 0.96]  # Cobalt
        elif i < 300:
            r = 0.25 + _seeded_rand(i, 3) * 0.55
            coords[i] = [r * np.sin(v) * np.cos(u), r * np.cos(v), r * np.sin(v) * np.sin(u)]
            base_colors[i] = [0.89, 0.89, 0.91]  # Zinc
        elif i < 400:
            r = 0.95 + _seeded_rand(i, 3) * 0.30
            ang = ((i - 300) / 100.0) * np.pi * 2.0
            coords[i] = [r * np.cos(ang), (_seeded_rand(i, 4) - 0.5) * 0.8, r * np.sin(ang)]
            base_colors[i] = [0.98, 0.45, 0.09]  # Orange
        else:
            q = (i - 400) // 25
            c = centers[min(q, 3)]
            coords[i] = [
                c[0] + (_seeded_rand(i, 3) - 0.5) * 0.42,
                c[1] + (_seeded_rand(i, 4) - 0.5) * 0.42,
                c[2] + (_seeded_rand(i, 5) - 0.5) * 0.42,
            ]
            base_colors[i] = [0.06, 0.73, 0.51]  # Emerald
    return coords, base_colors


def _greedy_action(cursor: np.ndarray, target: np.ndarray) -> int:
    delta = target - cursor
    best_a, best_dot = 0, -1e9
    for a, vec in ACTION_VECTORS.items():
        score = float(np.dot(delta, vec / (np.linalg.norm(vec) + 1e-9)))
        if score > best_dot:
            best_dot = score
            best_a = a
    return best_a


def generate_demo(out_dir: str = "docs/assets", n_steps: int = 60):
    os.makedirs(out_dir, exist_ok=True)
    b2.start_scope()
    env = OrganoidEnv(difficulty_stage=4, seed=42)
    env.set_curriculum_stage(4)
    env.reset(seed=42)

    coords, base_colors = _build_3d_coords(500)
    dt_sec = float(env.step_duration / b2.second)
    prev_spikes = int(len(env.spike_mon.t))

    frames = []
    trail = [env.cursor_pos.copy()]
    rates_hist, atp_hist = [], []
    all_t, all_i = [], []

    plt.rcParams.update({
        "figure.facecolor": "#09090b",
        "axes.facecolor": "#111114",
        "axes.edgecolor": "#27272a",
        "text.color": "#fafafa",
        "axes.labelcolor": "#a1a1aa",
        "xtick.color": "#a1a1aa",
        "ytick.color": "#a1a1aa",
        "font.size": 9,
    })

    for step in range(n_steps):
        action = _greedy_action(env.cursor_pos, env.target_pos)
        _, reward, term, trunc, _ = env.step(action)
        trail.append(env.cursor_pos.copy())

        cur_spikes = int(len(env.spike_mon.t))
        new_t = np.array(env.spike_mon.t[prev_spikes:] / b2.ms, dtype=np.float64)
        new_i = np.array(env.spike_mon.i[prev_spikes:], dtype=np.int32)
        prev_spikes = cur_spikes

        rate_hz = len(new_i) / (env.n_neurons * dt_sec)
        mean_atp = float(np.mean(env.neurons.E))
        rates_hist.append(rate_hz)
        atp_hist.append(mean_atp)
        all_t.extend(new_t.tolist())
        all_i.extend(new_i.tolist())

        # Render 2x2 frame
        fig = plt.figure(figsize=(11.2, 6.6), dpi=110)
        fig.suptitle(
            f"OrganoidEnv Live Closed-Loop Demo  |  Step {step+1}/{n_steps}  |  Rate: {rate_hz:.1f} Hz  |  Energy E: {mean_atp:.3f}",
            fontsize=11.5, fontweight="bold", color="#fafafa", y=0.97,
        )

        # 1. 3D Organoid Connectome
        ax3d = fig.add_subplot(2, 2, 1, projection="3d", facecolor="#09090b")
        ax3d.set_axis_off()
        ax3d.view_init(elev=18, azim=25 + step * 3.0)
        active_mask = np.zeros(500, dtype=bool)
        if len(new_i):
            active_mask[np.clip(new_i, 0, 499)] = True
        colors = base_colors.copy()
        colors[active_mask] = [1.0, 1.0, 1.0]
        sizes = np.where(active_mask, 32.0, 8.0)
        alphas = np.where(active_mask, 1.0, 0.45)
        ax3d.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colors, s=sizes, alpha=alphas, depthshade=False)
        ax3d.set_title("3D Organoid Connectome (500 Neurons)", fontsize=10, color="#e4e4e7", pad=2)

        # 2. 2D Navigation Arena
        ax2d = fig.add_subplot(2, 2, 2)
        for obs in env.obstacles:
            if obs["type"] == "circle":
                circ = plt.Circle(obs["pos"], obs["radius"], color="#ef4444", alpha=0.30, ec="#ef4444", lw=1.2)
                ax2d.add_patch(circ)
            elif obs["type"] == "box":
                x0, x1, y0, y1 = obs["bounds"]
                rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color="#ef4444", alpha=0.30, ec="#ef4444", lw=1.2)
                ax2d.add_patch(rect)
        goal_c = plt.Circle(env.target_pos, env.goal_radius, color="#10b981", alpha=0.30, ec="#10b981", lw=1.5)
        ax2d.add_patch(goal_c)
        tr = np.asarray(trail)
        ax2d.plot(tr[:, 0], tr[:, 1], color="#3b82f6", lw=1.8, alpha=0.85)
        ax2d.scatter([env.cursor_pos[0]], [env.cursor_pos[1]], color="#fafafa", s=48, zorder=5)
        ax2d.set_xlim(0, 1); ax2d.set_ylim(0, 1)
        ax2d.set_aspect("equal")
        ax2d.set_title("2D Closed-Loop Arena (Stage 4, r=0.05)", fontsize=10, color="#e4e4e7")

        # 3. Scrolling Spike Raster
        ax_ras = fig.add_subplot(2, 2, 3)
        t_now = (step + 1) * 100.0
        t_lo = max(0.0, t_now - 3000.0)
        t_arr = np.asarray(all_t)
        i_arr = np.asarray(all_i)
        m = (t_arr >= t_lo) & (t_arr <= t_now)
        if np.any(m):
            c_ras = np.where(i_arr[m] < 256, "#3b82f6",
                    np.where(i_arr[m] < 300, "#e4e4e7",
                    np.where(i_arr[m] < 400, "#f97316", "#10b981")))
            ax_ras.scatter(t_arr[m] / 1000.0, i_arr[m], c=c_ras, s=2.2, alpha=0.85, edgecolors="none")
        ax_ras.set_xlim(t_lo / 1000.0, max(3.0, t_now / 1000.0))
        ax_ras.set_ylim(0, 500)
        ax_ras.set_xlabel("Time (s)"); ax_ras.set_ylabel("Neuron Index")
        ax_ras.set_title("500-Neuron Spike Raster", fontsize=10, color="#e4e4e7")

        # 4. Firing Rate & Energy Trace
        ax_tr = fig.add_subplot(2, 2, 4)
        xs = np.arange(1, len(rates_hist) + 1) * 0.1
        ax_tr.axhspan(2.0, 20.0, color="#10b981", alpha=0.12, label="Target Band (2-20 Hz)")
        ax_tr.plot(xs, rates_hist, color="#3b82f6", lw=1.8, label="Rate (Hz)")
        ax_tr.set_xlim(0, max(3.0, n_steps * 0.1))
        ax_tr.set_ylim(0, 30.0)
        ax_tr.set_xlabel("Time (s)"); ax_tr.set_ylabel("Rate (Hz)", color="#3b82f6")
        ax_e = ax_tr.twinx()
        ax_e.plot(xs, atp_hist, color="#f59e0b", lw=1.8, label="Energy E(t)")
        ax_e.set_ylim(-0.05, 1.05)
        ax_e.set_ylabel("Energy E", color="#f59e0b")
        ax_tr.set_title("Population Rate & Energy Fatigue", fontsize=10, color="#e4e4e7")

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(buf.copy())
        plt.close(fig)

        if term or trunc:
            env.reset(seed=42 + step, options={"preserve_weights": True})
            trail = [env.cursor_pos.copy()]

    gif_path = os.path.join(out_dir, "organoidenv_demo.gif")
    mp4_path = os.path.join(out_dir, "organoidenv_demo.mp4")

    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=120, loop=0, optimize=True)

    h, w, _ = frames[0].shape
    writer = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Generated {gif_path} ({os.path.getsize(gif_path)} bytes) and {mp4_path} ({os.path.getsize(mp4_path)} bytes)")


if __name__ == "__main__":
    generate_demo()
