"""
FastAPI + WebSocket Interactive Platform Backend for OrganoidEnv.

Exposes:
  - Live closed-loop SNN simulation engine (REST + WebSocket streaming)
  - Real-time single-flag ablations, topology switching, and quadrant probes
  - Benchmark Explorer API backed by results/benchmarks/{stats_summary,stability_sweep,sensitivity_sweep}.json
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import brian2 as b2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
BENCH_DIR = os.path.join(ROOT_DIR, "results", "benchmarks")

TOPOLOGIES: Dict[str, List[Dict[str, Any]]] = {
    "T0_nominal": [
        {"type": "circle", "pos": [0.5, 0.5], "radius": 0.15},
        {"type": "box", "bounds": [0.2, 0.3, 0.4, 0.8]},
    ],
    "T1_dual_circle": [
        {"type": "circle", "pos": [0.40, 0.60], "radius": 0.12},
        {"type": "circle", "pos": [0.60, 0.40], "radius": 0.12},
    ],
    "T2_slanted_walls": [
        {"type": "box", "bounds": [0.35, 0.43, 0.25, 0.72]},
        {"type": "box", "bounds": [0.57, 0.65, 0.35, 0.88]},
    ],
    "T3_central_maze": [
        {"type": "circle", "pos": [0.45, 0.55], "radius": 0.16},
        {"type": "box", "bounds": [0.58, 0.80, 0.46, 0.54]},
    ],
    "T4_tight_corridor": [
        {"type": "box", "bounds": [0.25, 0.58, 0.56, 0.65]},
        {"type": "box", "bounds": [0.38, 0.72, 0.32, 0.41]},
    ],
}

# Action directions matching OrganoidEnv._apply_action (0..7)
ACTION_VECTORS = {
    0: np.array([0.0, 1.0]),    # Up
    1: np.array([0.0, -1.0]),   # Down
    2: np.array([-1.0, 0.0]),   # Left
    3: np.array([1.0, 0.0]),    # Right
    4: np.array([1.0, 1.0]),    # Up-Right
    5: np.array([-1.0, 1.0]),   # Up-Left
    6: np.array([1.0, -1.0]),   # Down-Right
    7: np.array([-1.0, -1.0]),  # Down-Left
}


class SimConfigRequest(BaseModel):
    use_metabolism: bool = True
    use_stabilizer: bool = True
    use_plasticity: bool = True
    use_pruning: bool = True
    use_dual_trace: bool = True
    random_snn: bool = False
    difficulty_stage: int = 4
    topology: str = "T0_nominal"
    seed: int = 42
    preserve_weights: bool = False
    spike_cost: float = 0.1
    tau_recovery_exc: float = 1000.0


class StepRequest(BaseModel):
    action: Optional[int] = None
    policy: str = "goal_seeker"  # "manual", "goal_seeker", "random"


def _serialize_obstacles(obstacles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for obs in obstacles:
        d = {"type": obs["type"]}
        if "pos" in obs:
            d["pos"] = [float(x) for x in obs["pos"]]
        if "radius" in obs:
            d["radius"] = float(obs["radius"])
        if "bounds" in obs:
            d["bounds"] = [float(x) for x in obs["bounds"]]
        out.append(d)
    return out


class SimulationSession:
    """Manages a live, reusable OrganoidEnv instance for interactive stepping and streaming."""

    def __init__(self):
        self.env = None
        self.config = SimConfigRequest()
        self.step_idx = 0
        self.episode_return = 0.0
        self.prev_spikes = 0
        self.trail: List[List[float]] = []
        self.last_reward = 0.0
        self.done = False

    def ensure_env(self, cfg: Optional[SimConfigRequest] = None, force_reset: bool = True):
        if cfg is not None:
            self.config = cfg
        if self.env is None:
            b2.start_scope()
            from organoid_rl.environment.core import OrganoidEnv

            self.env = OrganoidEnv(
                use_metabolism=self.config.use_metabolism,
                use_stabilizer=self.config.use_stabilizer,
                use_plasticity=self.config.use_plasticity,
                use_pruning=self.config.use_pruning,
                use_dual_trace=self.config.use_dual_trace,
                random_snn=self.config.random_snn,
                difficulty_stage=self.config.difficulty_stage,
                spike_cost=self.config.spike_cost,
                tau_recovery_exc=self.config.tau_recovery_exc,
                seed=self.config.seed,
            )
        else:
            # Update ablation flags live without rebuilding Brian2/C kernel
            self.env.use_metabolism = bool(self.config.use_metabolism)
            self.env.use_stabilizer = bool(self.config.use_stabilizer)
            self.env.use_plasticity = bool(self.config.use_plasticity)
            self.env.use_pruning = bool(self.config.use_pruning)
            self.env.use_dual_trace = bool(self.config.use_dual_trace)
            self.env.random_snn = bool(self.config.random_snn)
            self.env.spike_cost = float(self.config.spike_cost)
            if getattr(self.env, "_fast_runner", None) is not None:
                self.env._fast_runner.use_metabolism = bool(self.config.use_metabolism)
                self.env._fast_runner.spike_cost = float(self.config.spike_cost)

        self.env.set_curriculum_stage(int(self.config.difficulty_stage))
        if force_reset:
            opts = {"preserve_weights": True} if self.config.preserve_weights else None
            self.env.reset(seed=int(self.config.seed), options=opts)
            self._apply_topology(self.config.topology)
            self.step_idx = 0
            self.episode_return = 0.0
            self.last_reward = 0.0
            self.done = False
            self.prev_spikes = int(len(self.env.spike_mon.t))
            self.trail = [[float(self.env.cursor_pos[0]), float(self.env.cursor_pos[1])]]
        else:
            self._apply_topology(self.config.topology)

    def _apply_topology(self, topo_name: str):
        if topo_name in TOPOLOGIES and self.config.difficulty_stage >= 2:
            parsed = []
            for item in TOPOLOGIES[topo_name]:
                d = dict(item)
                if "pos" in d:
                    d["pos"] = np.array(d["pos"], dtype=np.float64)
                parsed.append(d)
            self.env.obstacles = parsed
            if hasattr(self.env, "_cached_prox"):
                self.env._cached_prox = self.env._compute_obstacle_proximity()

    def choose_policy_action(self, policy: str, manual_action: Optional[int] = None) -> int:
        if policy == "manual" and manual_action is not None:
            return int(np.clip(manual_action, 0, 7))
        if policy == "random":
            return int(np.random.randint(0, 8))
        # Default: greedy goal-seeker heuristic
        delta = self.env.target_pos - self.env.cursor_pos
        best_a = 0
        best_dot = -1e9
        for a, vec in ACTION_VECTORS.items():
            norm_v = vec / (np.linalg.norm(vec) + 1e-9)
            score = float(np.dot(delta, norm_v))
            if score > best_dot:
                best_dot = score
                best_a = a
        return best_a

    def step_once(self, policy: str = "goal_seeker", manual_action: Optional[int] = None) -> Dict[str, Any]:
        if self.env is None:
            self.ensure_env(force_reset=True)
        if self.done:
            self.ensure_env(force_reset=True)

        action = self.choose_policy_action(policy, manual_action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_idx += 1
        self.last_reward = float(reward)
        self.episode_return += float(reward)
        self.done = bool(terminated or truncated)
        self.trail.append([float(self.env.cursor_pos[0]), float(self.env.cursor_pos[1])])
        if len(self.trail) > 120:
            self.trail = self.trail[-120:]

        return self.snapshot(action=action, info=info)

    def probe_quadrant(self, quadrant: int) -> Dict[str, Any]:
        """Injects a +25.0 directional probe into motor quadrant (0=Up, 1=Down, 2=Left, 3=Right)."""
        if self.env is None:
            self.ensure_env(force_reset=True)
        return self.step_once(policy="manual", manual_action=int(np.clip(quadrant, 0, 3)))

    def snapshot(self, action: Optional[int] = None, info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        env = self.env
        cur_spikes = int(len(env.spike_mon.t))
        new_t = np.array(env.spike_mon.t[self.prev_spikes :] / b2.ms, dtype=np.float64)
        new_i = np.array(env.spike_mon.i[self.prev_spikes :], dtype=np.int32)
        total_step_spikes = int(len(new_i))
        self.prev_spikes = cur_spikes

        # Subsample spikes for smooth WebSocket rendering if extremely dense (>1500/step)
        if len(new_i) > 1500:
            idx = np.linspace(0, len(new_i) - 1, 1500, dtype=int)
            new_t = new_t[idx]
            new_i = new_i[idx]

        dt_sec = float(env.step_duration / b2.second)
        pop_rate_hz = float(total_step_spikes / (env.n_neurons * dt_sec))

        # Motor quadrant firing rates (Hz)
        motor_slices = [env.motor_up, env.motor_down, env.motor_left, env.motor_right]
        motor_rates = []
        for lo, hi in motor_slices:
            cnt = int(np.sum((new_i >= lo) & (new_i < hi)))
            motor_rates.append(round(cnt / (25.0 * dt_sec), 2))

        e_arr = np.array(env.neurons.E, dtype=np.float64)
        mean_atp = float(np.mean(e_arr))
        exhausted_frac = float(np.mean(e_arr <= 0.0))

        sdm_active = len(set(int(x) for x in getattr(env, "_cached_recent_spikes_i", []) if x < env.n_sdm))
        sdm_pct = round(100.0 * sdm_active / max(1, env.n_sdm), 2)

        w_means = [float(np.mean(np.array(S.w))) for S in env.synapses[:-1]] if len(env.synapses) > 1 else [0.0]
        mean_w = float(np.mean(w_means))

        counters = dict(getattr(env, "gar_state_counters", {"seizure_clamps": 0, "coma_kicks": 0, "nan_resets": 0}))
        dist = float(np.linalg.norm(env.cursor_pos - env.target_pos))

        return {
            "step": self.step_idx,
            "action": action,
            "done": self.done,
            "reward": round(self.last_reward, 3),
            "episode_return": round(self.episode_return, 3),
            "distance": round(dist, 4),
            "goal_radius": float(env.goal_radius),
            "cursor": [round(float(env.cursor_pos[0]), 4), round(float(env.cursor_pos[1]), 4)],
            "target": [round(float(env.target_pos[0]), 4), round(float(env.target_pos[1]), 4)],
            "trail": self.trail,
            "obstacles": _serialize_obstacles(env.obstacles),
            "pop_rate_hz": round(pop_rate_hz, 2),
            "mean_atp": round(mean_atp, 4),
            "exhausted_frac": round(exhausted_frac, 4),
            "sdm_occupancy_pct": sdm_pct,
            "mean_weight": round(mean_w, 3),
            "motor_rates_hz": {
                "up": motor_rates[0],
                "down": motor_rates[1],
                "left": motor_rates[2],
                "right": motor_rates[3],
            },
            "gar_counters": counters,
            "spikes": {
                "t_ms": [round(float(t), 2) for t in new_t],
                "i": [int(i) for i in new_i],
                "total_step_spikes": total_step_spikes,
            },
            "config": self.config.model_dump(),
        }


class SessionManager:
    """Thread-safe, capacity-bounded manager for isolated per-user SimulationSession instances."""

    def __init__(self, max_sessions: int = 32, ttl_sec: float = 1800.0):
        self.max_sessions = max_sessions
        self.ttl_sec = ttl_sec
        self._sessions: Dict[str, SimulationSession] = {}
        self._last_used: Dict[str, float] = {}
        self._lock = threading.Lock()

    def _evict_stale_locked(self, now: float):
        expired = [sid for sid, ts in self._last_used.items() if now - ts > self.ttl_sec]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._last_used.pop(sid, None)
        while len(self._sessions) >= self.max_sessions:
            oldest_sid = min(self._last_used, key=self._last_used.get)
            self._sessions.pop(oldest_sid, None)
            self._last_used.pop(oldest_sid, None)

    def get_or_create(self, session_id: Optional[str] = None) -> tuple[str, SimulationSession]:
        sid = session_id or "default"
        now = time.monotonic()
        with self._lock:
            if sid in self._sessions:
                self._last_used[sid] = now
                return sid, self._sessions[sid]
            self._evict_stale_locked(now)
            sess = SimulationSession()
            self._sessions[sid] = sess
            self._last_used[sid] = now
            return sid, sess

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "active_sessions": len(self._sessions),
                "max_sessions": self.max_sessions,
                "ttl_sec": self.ttl_sec,
            }


def create_app(max_sessions: int = 32) -> FastAPI:
    app = FastAPI(title="OrganoidEnv Interactive Platform", version="1.2.0")
    manager = SessionManager(max_sessions=max_sessions)

    if os.path.isdir(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    def serve_index():
        index_path = os.path.join(STATIC_DIR, "index.html")
        if not os.path.isfile(index_path):
            return HTMLResponse("<h1>OrganoidEnv Platform UI not found.</h1>", status_code=404)
        return FileResponse(index_path)

    @app.get("/api/health")
    def health():
        from organoid_rl.environment.fast_snn import load_snn_kernel

        return {
            "status": "ok",
            "c_kernel_loaded": load_snn_kernel() is not None,
            "topologies": list(TOPOLOGIES.keys()),
            "sessions": manager.stats(),
        }

    @app.get("/api/sessions")
    def sessions_info():
        return manager.stats()

    @app.get("/api/benchmarks")
    def get_benchmarks():
        summary_csv = os.path.join(BENCH_DIR, "summary_table.csv")
        stats_json = os.path.join(BENCH_DIR, "stats_summary.json")
        if not os.path.isfile(stats_json):
            raise HTTPException(status_code=404, detail="stats_summary.json not found")
        with open(stats_json, encoding="utf-8") as f:
            stats = json.load(f)
        rows = []
        if os.path.isfile(summary_csv):
            with open(summary_csv, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        return {"table": rows, "stats": stats}

    @app.get("/api/sweeps")
    def get_sweeps():
        stab_path = os.path.join(BENCH_DIR, "stability_sweep.json")
        sens_path = os.path.join(BENCH_DIR, "sensitivity_sweep.json")
        out: Dict[str, Any] = {}
        if os.path.isfile(stab_path):
            with open(stab_path, encoding="utf-8") as f:
                out["stability_sweep"] = json.load(f)
        if os.path.isfile(sens_path):
            with open(sens_path, encoding="utf-8") as f:
                out["sensitivity_sweep"] = json.load(f)
        return out

    @app.post("/api/sim/configure")
    def configure_sim(cfg: SimConfigRequest, session_id: str = Query(default="default")):
        _, session = manager.get_or_create(session_id)
        session.ensure_env(cfg=cfg, force_reset=True)
        return session.snapshot()

    @app.post("/api/sim/step")
    def step_sim(req: StepRequest, session_id: str = Query(default="default")):
        _, session = manager.get_or_create(session_id)
        return session.step_once(policy=req.policy, manual_action=req.action)

    @app.websocket("/api/ws/sim")
    async def ws_sim(websocket: WebSocket, session_id: Optional[str] = Query(default=None)):
        await websocket.accept()
        sid = session_id or str(uuid.uuid4())
        _, session = manager.get_or_create(sid)
        if session.env is None:
            await asyncio.to_thread(session.ensure_env, None, True)
        await websocket.send_json(session.snapshot())

        running = False
        policy = "goal_seeker"
        interval_sec = 0.12

        try:
            while True:
                try:
                    msg = await asyncio.wait_for(websocket.receive_json(), timeout=interval_sec if running else 30.0)
                    cmd = msg.get("cmd")
                    if cmd == "play":
                        running = True
                        policy = msg.get("policy", policy)
                        interval_sec = float(msg.get("interval_ms", 120)) / 1000.0
                    elif cmd == "pause":
                        running = False
                    elif cmd == "step":
                        running = False
                        frame = await asyncio.to_thread(
                            session.step_once,
                            msg.get("policy", policy),
                            msg.get("action"),
                        )
                        await websocket.send_json(frame)
                    elif cmd == "probe":
                        running = False
                        frame = await asyncio.to_thread(session.probe_quadrant, int(msg.get("quadrant", 0)))
                        await websocket.send_json(frame)
                    elif cmd == "configure":
                        cfg = SimConfigRequest(**msg.get("config", {}))
                        force_reset = bool(msg.get("reset", True))
                        await asyncio.to_thread(session.ensure_env, cfg, force_reset)
                        await websocket.send_json(session.snapshot())
                    elif cmd == "set_policy":
                        policy = msg.get("policy", policy)
                except asyncio.TimeoutError:
                    pass

                if running:
                    frame = await asyncio.to_thread(session.step_once, policy)
                    await websocket.send_json(frame)
        except WebSocketDisconnect:
            pass

    return app


app = create_app()
