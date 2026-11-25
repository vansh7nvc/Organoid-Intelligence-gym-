"""Tests for the OrganoidEnv Interactive Platform (FastAPI + SimulationSession)."""

from organoid_rl.platform.server import SessionManager, SimConfigRequest, SimulationSession, StepRequest, create_app


def test_platform_routes_and_data_endpoints():
    app = create_app()
    routes = {r.path: r for r in app.routes}
    assert "/" in routes
    assert "/api/health" in routes
    assert "/api/benchmarks" in routes
    assert "/api/sweeps" in routes
    assert "/api/sim/configure" in routes
    assert "/api/sim/step" in routes
    assert "/api/ws/sim" in routes

    health_fn = routes["/api/health"].endpoint
    h = health_fn()
    assert h["status"] == "ok"
    assert "T4_tight_corridor" in h["topologies"]

    bench_fn = routes["/api/benchmarks"].endpoint
    b_data = bench_fn()
    assert len(b_data["table"]) >= 15
    assert "c8_no_gar_no_met" in b_data["stats"]["conditions"]

    sweeps_fn = routes["/api/sweeps"].endpoint
    s_data = sweeps_fn()
    assert "stability_sweep" in s_data
    assert "sensitivity_sweep" in s_data


def test_simulation_session_configure_step_and_probe():
    session = SimulationSession()
    session.ensure_env(
        cfg=SimConfigRequest(
            use_metabolism=True,
            use_stabilizer=True,
            topology="T4_tight_corridor",
            difficulty_stage=4,
            seed=42,
        ),
        force_reset=True,
    )
    snap0 = session.snapshot()
    assert snap0["step"] == 0
    assert len(snap0["obstacles"]) == 2
    assert snap0["goal_radius"] == 0.05

    snap1 = session.step_once(policy="goal_seeker")
    assert snap1["step"] == 1
    assert "spikes" in snap1 and "t_ms" in snap1["spikes"]
    assert "motor_rates_hz" in snap1

    # Live ablation toggle without full reset
    session.ensure_env(
        cfg=SimConfigRequest(use_metabolism=False, use_stabilizer=False, topology="T4_tight_corridor"),
        force_reset=False,
    )
    assert session.env.use_metabolism is False
    assert session.env.use_stabilizer is False

    probe_snap = session.probe_quadrant(0)
    assert probe_snap["action"] == 0


def test_session_manager_isolation_and_eviction():
    mgr = SessionManager(max_sessions=2, ttl_sec=3600.0)
    _, s1 = mgr.get_or_create("alice")
    _, s2 = mgr.get_or_create("bob")
    assert s1 is not s2
    assert mgr.stats()["active_sessions"] == 2

    # Capacity-bounded eviction when adding a 3rd session
    _, s3 = mgr.get_or_create("carol")
    assert s3 is not s1 and s3 is not s2
    assert mgr.stats()["active_sessions"] == 2
