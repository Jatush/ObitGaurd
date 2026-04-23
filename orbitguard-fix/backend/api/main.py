"""OrbitGuard API v3.1 — fully async, SGP4-propagated real TLE debris."""
import asyncio, time, json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from core.simulation import (
    SimulationConfig, DebrisObject, run_simulation,
    keplerian_to_state, state_to_keplerian, orbit_ring,
    compute_pc, risk_level, R_EARTH, generate_synthetic_debris,
    optimize_maneuver_rl, _propagate, _state_at, RL_MODEL_METADATA
)
from services.tle_service import tle_service

# ── PPO agent (lazy-loaded on first maneuver call) ────────────────────────────
_ppo_agent   = None
_ppo_metrics = None

def _get_ppo_agent():
    global _ppo_agent, _ppo_metrics
    if _ppo_agent is None:
        try:
            from core.ppo_agent import get_or_train_agent
            _ppo_agent, _ppo_metrics = get_or_train_agent(
                n_episodes=300, force_retrain=False, verbose=True)
        except Exception as e:
            print(f"[PPO] Warning: {e}")
    return _ppo_agent, _ppo_metrics

POOL = ThreadPoolExecutor(max_workers=4)
_reports: Dict[str, dict] = {}
_ws_clients: List[WebSocket] = []
REPORTS_STORE_PATH = Path(__file__).resolve().parents[1] / "data" / "reports_store.json"
CLOSE_PASS_WARN_KM = 1.0

def _load_reports_store() -> Dict[str, dict]:
    try:
        if not REPORTS_STORE_PATH.exists():
            return {}
        raw = json.loads(REPORTS_STORE_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        out: Dict[str, dict] = {}
        for k, v in raw.items():
            if isinstance(v, dict):
                out[str(k).upper()] = v
        return out
    except Exception as exc:
        print(f"[API] Could not load reports store: {exc}")
        return {}

def _save_reports_store() -> None:
    try:
        REPORTS_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Keep recent entries bounded for fast startup.
        items = list(_reports.items())[-200:]
        REPORTS_STORE_PATH.write_text(
            json.dumps({k: v for k, v in items}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"[API] Could not save reports store: {exc}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    print("🛸 OrbitGuard API v3.1 starting…")
    _reports.update(_load_reports_store())
    if _reports:
        print(f"   Restored reports: {len(_reports)}")
    n = await loop.run_in_executor(POOL, tle_service.fetch_live)
    print(f"   TLE catalog: {len(tle_service.get_all())} objects ({n} from live)")
    yield
    _save_reports_store()
    POOL.shutdown(wait=False)

app = FastAPI(title="OrbitGuard API", version="3.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(RequestValidationError)
async def val_err(req, exc):
    errs = []
    for e in exc.errors():
        field = " -> ".join(str(x) for x in e["loc"])
        errs.append({"field": field, "message": e["msg"], "input": str(e.get("input",""))})
    return JSONResponse(
        status_code=422,
        content={"detail": errs,
            "hint": "duration_hours max 168, max_dv_ms max 500, isp_s max 10000, altitude_km max 60000"})

# ── Schemas ───────────────────────────────────────────────────────────────────
class ElemsIn(BaseModel):
    altitude_km:      float = Field(550.,  ge=160,  le=60000)
    inclination_deg:  float = Field(53.,   ge=0,    le=360)
    eccentricity:     float = Field(0.001, ge=0,    le=0.99)
    raan_deg:         float = Field(0.,    ge=0,   le=360)
    argp_deg:         float = Field(0.,    ge=0,   le=360)
    true_anomaly_deg: float = Field(0.,    ge=0,   le=360)

class TLEIn(BaseModel):
    name: str; line1: str; line2: str

class SimReq(BaseModel):
    satellite_name:        str   = "MY-SAT-1"
    tle:                   Optional[TLEIn]   = None
    elements:              Optional[ElemsIn] = None
    duration_hours:        float = Field(12.,  ge=1,    le=168)   # up to 7 days
    max_dv_ms:             float = Field(5.,   ge=0.1,  le=500)   # up to 500 m/s
    satellite_mass_kg:     float = Field(300., ge=1,    le=100000)
    satellite_isp_s:       float = Field(220., ge=10,   le=10000) # ion engines
    screening_distance_km: float = Field(10.,  ge=0.1,  le=500)
    use_live_tle:          bool  = True
    use_synthetic:         bool  = False
    scan_all_tle:          bool  = False   # scan every TLE object in catalog
    n_synthetic:           int   = Field(8, ge=1,    le=99999)
    max_live_debris:       int   = Field(200, ge=1,   le=5000)
    norad_ids:             Optional[List[int]] = None
    force_collision_demo:  bool  = False   # inject a guaranteed-collision debris object
    client_request_id:     Optional[str] = None

class ConjReq(BaseModel):
    primary: ElemsIn; secondary: ElemsIn
    duration_hours: float = Field(24., ge=1, le=72)

class ManReq(BaseModel):
    primary:           ElemsIn; debris: ElemsIn
    tca_hours:         float = Field(6.,  ge=0.1,  le=168)
    duration_hours:    float = Field(24., ge=1,    le=168)
    max_dv_ms:         float = Field(5.,  ge=0.1,  le=500)
    satellite_mass_kg: float = Field(300.,ge=1,    le=100000)
    isp_s:             float = Field(220.,ge=10,   le=10000)

# ── Helpers ───────────────────────────────────────────────────────────────────
def elems_to_state(e: ElemsIn) -> np.ndarray:
    return keplerian_to_state(R_EARTH+e.altitude_km*1000, e.eccentricity,
                               e.inclination_deg, e.raan_deg, e.argp_deg, e.true_anomaly_deg)

def tle_obj_to_debris(obj, jd: float, fr: float) -> Optional[DebrisObject]:
    """
    Use SGP4 to get real current ECI position of TLE object.
    Falls back to Keplerian approximation if SGP4 fails.
    """
    state = obj.get_eci_state(jd, fr)
    if state is None:
        # Fallback: approximate from TLE-derived orbital elements
        try:
            state = keplerian_to_state(
                R_EARTH + obj.altitude_km*1000,
                obj.eccentricity, obj.inclination_deg,
                obj.raan_deg, obj.argp_deg, obj.mean_anomaly_deg
            )
        except Exception:
            return None
    return DebrisObject(
        id=f"TLE-{obj.norad_id}", name=obj.name, norad_id=obj.norad_id,
        state=state, object_type=obj.object_type, source="tle_sgp4"
    )

async def broadcast(msg: dict):
    dead = []
    for ws in _ws_clients:
        try: await ws.send_json(msg)
        except: dead.append(ws)
    for ws in dead: _ws_clients.remove(ws)

def _state_from_report_primary(primary_state: dict) -> Optional[np.ndarray]:
    try:
        return np.array([
            float(primary_state["x_m"]), float(primary_state["y_m"]), float(primary_state["z_m"]),
            float(primary_state["vx_ms"]), float(primary_state["vy_ms"]), float(primary_state["vz_ms"]),
        ], dtype=float)
    except Exception:
        return None

def _post_rescreen_compute(report: dict, debris_list: List[DebrisObject], screening_distance_km: float, hard_body_radius_m: float):
    """Compute post-burn re-screening for all maneuver plans (background)."""
    plans = list(report.get("maneuver_plans") or [])
    if not plans:
        return {
            "post_maneuver_rescreen_status": "not_applicable",
            "all_plans_post_screening": [],
            "post_maneuver_screening": None,
        }

    s0p = _state_from_report_primary(report.get("primary_state") or {})
    if s0p is None:
        return {
            "post_maneuver_rescreen_status": "failed",
            "post_maneuver_rescreen_error": "primary state unavailable",
        }

    dur = float(report.get("simulation_duration_hours", 12.0)) * 3600.0
    pre_ids = {c.get("debris_id") for c in (report.get("conjunctions") or [])}
    max_recheck = min(len(debris_list), 250)
    if len(debris_list) > max_recheck:
        r0 = np.linalg.norm(s0p[:3])
        debris_sorted = sorted(debris_list, key=lambda d: abs(np.linalg.norm(d.state[:3]) - r0))
        debris_check = debris_sorted[:max_recheck]
    else:
        debris_check = debris_list

    summaries = []
    from core.simulation import _propagate_burn, _propagate, compute_pc, risk_level
    for p in plans:
        burn_at = float(p.get("burn_time_s", 0.0))
        dv_eci = np.array(p.get("delta_v_vector_eci", [0, 0, 0]), dtype=float)
        _, pos_p = _propagate_burn(s0p, dur, 140, burn_at, dv_eci, 60.0)
        t_arr = np.linspace(0, dur, len(pos_p))
        events = []
        for deb in debris_check:
            _, pos_d = _propagate(deb.state, dur, 140, 60.0)
            nd = min(len(pos_p), len(pos_d))
            if nd <= 1:
                continue
            d = np.linalg.norm(pos_p[:nd] - pos_d[:nd], axis=1)
            md = float(np.min(d))
            if md > screening_distance_km * 1000.0:
                continue
            mi = int(np.argmin(d))
            pc = compute_pc(md)
            events.append({
                "debris_id": deb.id,
                "debris_name": deb.name,
                "miss_distance_km": round(md / 1000.0, 5),
                "tca_hours": round(float(t_arr[mi]) / 3600.0, 4),
                "pc_scientific": f"{pc:.4e}",
                "risk_level": risk_level(pc),
                "collision_detected": md <= hard_body_radius_m,
                "is_new_vs_pre_maneuver": deb.id not in pre_ids,
            })
        events.sort(key=lambda e: float(e["pc_scientific"]), reverse=True)
        counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for e in events:
            counts[e["risk_level"]] += 1
        s = {
            "strategy": p.get("strategy", p.get("direction", "unknown")),
            "debris_rechecked": len(debris_check),
            "total_events": len(events),
            "new_events_vs_pre_maneuver": sum(1 for e in events if e["is_new_vs_pre_maneuver"]),
            "critical_count": counts["CRITICAL"],
            "high_count": counts["HIGH"],
            "medium_count": counts["MEDIUM"],
            "low_count": counts["LOW"],
            "top_events": events[:20],
        }
        p["post_screening_summary"] = {
            "debris_rechecked": s["debris_rechecked"],
            "total_events": s["total_events"],
            "new_events_vs_pre_maneuver": s["new_events_vs_pre_maneuver"],
            "critical_count": s["critical_count"],
            "high_count": s["high_count"],
        }
        summaries.append(s)

    rec = report.get("recommended_plan") or {}
    rec_key = rec.get("strategy", rec.get("direction"))
    rec_sum = next((s for s in summaries if s["strategy"] == rec_key), summaries[0] if summaries else None)
    return {
        "post_maneuver_rescreen_status": "done",
        "all_plans_post_screening": summaries,
        "post_maneuver_screening": rec_sum,
        "post_maneuver_rescreen_done": True,
    }

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/")
def root(): return {"service":"OrbitGuard API v3.1","status":"ok"}

@app.get("/api/health")
def health(): return {"status":"healthy","ts":time.time(),"tle_objects":len(tle_service.get_all())}

@app.get("/api/stats")
def stats(): return {**tle_service.get_summary(),"stored_reports":len(_reports)}

@app.get("/api/rl_model")
def get_rl_model(): return RL_MODEL_METADATA

# ── TLE catalog endpoints removed from public API ─────────────────────────────

# ── Main simulation ───────────────────────────────────────────────────────────
@app.post("/api/simulate")
async def run_sim(req: SimReq):
    cfg = SimulationConfig(
        satellite_name=req.satellite_name,
        duration_hours=req.duration_hours, max_dv_ms=req.max_dv_ms,
        satellite_mass_kg=req.satellite_mass_kg, satellite_isp_s=req.satellite_isp_s,
        screening_distance_km=req.screening_distance_km,
    )
    if req.tle:
        cfg.satellite_tle_line1 = req.tle.line1
        cfg.satellite_tle_line2 = req.tle.line2
    elif req.elements:
        e = req.elements
        cfg.altitude_km=e.altitude_km; cfg.inclination_deg=e.inclination_deg
        cfg.eccentricity=e.eccentricity; cfg.raan_deg=e.raan_deg
        cfg.argp_deg=e.argp_deg; cfg.true_anomaly_deg=e.true_anomaly_deg

    loop = asyncio.get_event_loop()

    # Build primary state
    if req.tle:
        from core.simulation import _state_from_tle
        s0p = await loop.run_in_executor(POOL, _state_from_tle, req.tle.line1, req.tle.line2)
        k = state_to_keplerian(s0p)
        cfg.altitude_km = k.get("altitude_km", 550)
        cfg.inclination_deg = k.get("inclination_deg", 53)
    else:
        elems = req.elements or ElemsIn()
        s0p = elems_to_state(elems)

    # Get current JD for SGP4
    jd, fr = tle_service.get_current_jd()

    # Build debris list
    debris: List[DebrisObject] = []

    if req.use_synthetic or not req.use_live_tle:
        synth_n = int(req.n_synthetic or 8)
        synth = await loop.run_in_executor(POOL, generate_synthetic_debris, s0p, synth_n, 42)
        debris += synth

    if req.use_live_tle:
        if req.norad_ids:
            for nid in req.norad_ids[:60]:
                o = tle_service.get_by_norad(nid)
                if o:
                    d = tle_obj_to_debris(o, jd, fr)
                    if d: debris.append(d)
        else:
            from services.tle_service import TLEObject
            dummy = TLEObject(
                norad_id=99999, name="TARGET", line1="", line2="",
                altitude_km=cfg.altitude_km, inclination_deg=cfg.inclination_deg,
                eccentricity=0.001, period_min=95,
                raan_deg=cfg.raan_deg, argp_deg=0, mean_anomaly_deg=0
            )
            # scan_all: screen entire TLE catalog; otherwise nearby by orbit
            if req.scan_all_tle:
                nearby = tle_service.get_all()
                print(f"[API] FULL CATALOG SCAN: {len(nearby)} objects")
            else:
                max_n = min(req.max_live_debris, 5000)
                nearby = tle_service.get_nearby_objects(dummy, alt_band_km=120, inc_band_deg=50, max_results=max_n)
                print(f"[API] Nearby scan: {len(nearby)} objects (max={max_n})")
            for o in nearby:
                d = tle_obj_to_debris(o, jd, fr)
                if d: debris.append(d)

            if len(debris) < 4:  # always have some real objects
                for o in tle_service.get_all()[:80]:
                    d = tle_obj_to_debris(o, jd, fr)
                    if d: debris.append(d)

    if not debris:
        debris = await loop.run_in_executor(POOL, generate_synthetic_debris, s0p, 8, 42)

    # Optional: inject a guaranteed collision object (same initial state as primary)
    if req.force_collision_demo:
        debris.append(DebrisObject(
            id="DEMO-COLLISION",
            name="FORCED-COLLISION-DEBRIS",
            norad_id=None,
            state=s0p.copy(),
            mass_kg=1.0,
            cross_section_m2=0.01,
            object_type="DEBRIS",
            source="demo_forced_collision",
        ))

    # Deduplicate
    seen = set(); unique = []
    for d in debris:
        if d.id not in seen: seen.add(d.id); unique.append(d)
    debris = unique

    # Progress → WS
    req_id = (req.client_request_id or "").strip() or None
    async def _bcast(pct, msg):
        await broadcast({"type":"progress","pct":pct,"msg":msg,"request_id":req_id})
    def progress_cb(pct, msg=""):
        asyncio.run_coroutine_threadsafe(_bcast(pct, msg), loop)

    # Run in thread pool
    report = await loop.run_in_executor(POOL, run_simulation, cfg, debris, progress_cb)
    report["post_maneuver_rescreen_status"] = "pending" if report.get("recommended_plan") else "not_applicable"
    report["post_maneuver_rescreen_done"] = False
    report["all_plans_post_screening"] = []
    report["post_maneuver_screening"] = None
    _reports[report["sim_id"]] = report
    _save_reports_store()
    # Run expensive post-burn re-screen in background after main simulation response.
    if report.get("recommended_plan"):
        sim_id = report["sim_id"]
        async def _bg_rescreen():
            try:
                res = await loop.run_in_executor(
                    POOL,
                    _post_rescreen_compute,
                    dict(_reports.get(sim_id, {})),
                    debris,
                    float(cfg.screening_distance_km),
                    float(cfg.hard_body_radius_m),
                )
                if sim_id in _reports:
                    _reports[sim_id].update(res or {})
                    _save_reports_store()
            except Exception as exc:
                if sim_id in _reports:
                    _reports[sim_id]["post_maneuver_rescreen_status"] = "failed"
                    _reports[sim_id]["post_maneuver_rescreen_error"] = str(exc)
                    _save_reports_store()
        asyncio.create_task(_bg_rescreen())
    await broadcast({"type":"simulation_complete","sim_id":report["sim_id"],
                      "critical":report["critical_count"],"high":report["high_count"],
                      "request_id":req_id})
    return report

# ── Demo ──────────────────────────────────────────────────────────────────────
@app.get("/api/demo")
async def demo():
    loop = asyncio.get_event_loop()
    s0p = keplerian_to_state(R_EARTH+550000, 0.001, 53, 0, 0, 0)
    debris = await loop.run_in_executor(POOL, generate_synthetic_debris, s0p, 8, 1234)
    cfg = SimulationConfig(satellite_name="DEMO-SAT", duration_hours=12,
                            altitude_km=550, inclination_deg=53)
    report = await loop.run_in_executor(POOL, run_simulation, cfg, debris, None)
    _reports[report["sim_id"]] = report
    _save_reports_store()
    return report

# ── Reports ───────────────────────────────────────────────────────────────────
@app.get("/api/reports")
def list_reports(limit:int=20):
    items = list(_reports.items())[-limit:]
    return {"reports":[{"sim_id":k,"created_at":v.get("created_at"),
                          "primary_name":v.get("primary_name"),
                          "n_conjunctions":len(v.get("conjunctions",[])),
                          "critical":v.get("critical_count",0)}
                         for k,v in reversed(items)]}

@app.get("/api/reports/{sim_id}")
def get_report(sim_id:str):
    r = _reports.get(sim_id.upper())
    if not r: raise HTTPException(404,"Not found")
    return r

# ── Quick tools ───────────────────────────────────────────────────────────────
@app.post("/api/conjunction")
async def quick_conjunction(req: ConjReq):
    loop = asyncio.get_event_loop()
    def _compute():
        sp=elems_to_state(req.primary); ss=elems_to_state(req.secondary)
        dur=req.duration_hours*3600
        _,pp=_propagate(sp,dur,400,60.); _,dp=_propagate(ss,dur,400,60.)
        n=min(len(pp),len(dp)); dists=np.linalg.norm(pp[:n]-dp[:n],axis=1)
        mi=int(np.argmin(dists)); md=float(dists[mi]); tca=dur*mi/n
        sp_tca=_state_at(sp,tca,90.); ss_tca=_state_at(ss,tca,90.)
        rv=float(np.linalg.norm(sp_tca[3:]-ss_tca[3:]))
        pc=compute_pc(md); step=max(1,n//200); t_arr=np.linspace(0,dur,n)
        miss_km = round(md/1000, 5)
        action_required = (pc > 1e-4) or (miss_km <= CLOSE_PASS_WARN_KM)
        return {"miss_distance_m":round(md,1),"miss_distance_km":miss_km,
                "tca_seconds":round(tca,1),"tca_hours":round(tca/3600,3),
                "relative_velocity_ms":round(rv,2),"relative_velocity_kms":round(rv/1000,4),
                "probability_of_collision":pc,"pc_scientific":f"{pc:.4e}",
                "risk_level":risk_level(pc),"action_required":action_required,
                "close_pass_warn_km": CLOSE_PASS_WARN_KM,
                "pc_action_threshold": 1e-4,
                "timeline":{"times_h":(t_arr[::step]/3600).tolist(),
                             "distances_km":(dists[::step]/1000).tolist()}}
    return await loop.run_in_executor(POOL, _compute)

@app.post("/api/maneuver")
async def compute_maneuver(req: ManReq):
    loop = asyncio.get_event_loop()
    def _compute():
        agent, agent_metrics = _get_ppo_agent()
        sp  = elems_to_state(req.primary)
        sd  = elems_to_state(req.debris)
        dur = req.duration_hours * 3600
        _, pp = _propagate(sp, dur, 300, 60.)
        _, dp = _propagate(sd, dur, 300, 60.)
        n  = min(len(pp), len(dp))
        nm = float(np.min(np.linalg.norm(pp[:n] - dp[:n], axis=1)))
        # optimize_maneuver_rl tries the real PPO agent first (via simulation.py)
        plans = optimize_maneuver_rl(
            sp, sd, req.tca_hours * 3600, dur,
            req.max_dv_ms, req.isp_s, req.satellite_mass_kg, nm)
        # Enrich model info with live training metrics
        model_info = dict(RL_MODEL_METADATA)
        if agent is not None and agent_metrics:
            tm = agent_metrics.get("task_metrics", {})
            tr = agent_metrics.get("training", {})
            md_m = agent_metrics.get("miss_distance", {})
            ev = agent_metrics.get("evaluation", {})
            model_info.update({
                "ppo_trained":      True,
                "ppo_framework":    "PyTorch",
                "ppo_episodes":     tr.get("n_episodes", 0),
                "ppo_success_rate": tm.get("success_rate", 0),
                "ppo_collision_rate": tm.get("collision_rate", 0),
                "ppo_mean_miss_km": md_m.get("mean_km", 0),
                "ppo_eval_success": ev.get("success_rate", 0),
            })
        else:
            model_info["ppo_trained"] = False
        return {
            "nominal_miss_m":  round(nm, 1),
            "nominal_miss_km": round(nm / 1000, 5),
            "nominal_pc":      compute_pc(nm),
            "nominal_risk":    risk_level(compute_pc(nm)),
            "plans":           plans,
            "recommended":     plans[0] if plans else None,
            "rl_model_info":   model_info,
        }
    return await loop.run_in_executor(POOL, _compute)


# ── PPO endpoints ─────────────────────────────────────────────────────────────
@app.get("/api/ppo/metrics")
def ppo_metrics_endpoint():
    _, metrics = _get_ppo_agent()
    if not metrics:
        raise HTTPException(status_code=404, detail="No trained PPO model. POST /api/ppo/train first.")
    return metrics

@app.post("/api/ppo/train")
async def train_ppo(episodes: int = 300, force: bool = False):
    loop = asyncio.get_event_loop()
    def _train():
        from core.ppo_agent import get_or_train_agent
        global _ppo_agent, _ppo_metrics
        _ppo_agent, _ppo_metrics = get_or_train_agent(
            n_episodes=max(10, min(episodes, 2000)),
            force_retrain=force, verbose=True)
        return _ppo_metrics
    return await loop.run_in_executor(POOL, _train)

@app.get("/api/ppo/info")
def ppo_info():
    _, metrics = _get_ppo_agent()
    return {
        "model": "PPO (Proximal Policy Optimization) - PyTorch",
        "state_dim": 9, "action_dim": 6,
        "actions": ["prograde","retrograde","radial+","radial-","normal+","normal-"],
        "architecture": {
            "actor":  "MLP 9->256->256->128->6 (tanh, Categorical)",
            "critic": "MLP 9->256->256->128->1 (tanh, linear)",
        },
        "training_metrics": metrics,
    }

# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept(); _ws_clients.append(ws)
    try:
        await ws.send_json({"type":"connected","ts":time.time()})
        while True:
            d = await ws.receive_json()
            if d.get("type") == "ping":
                await ws.send_json({"type":"pong","ts":time.time()})
    except WebSocketDisconnect:
        if ws in _ws_clients: _ws_clients.remove(ws)
    except Exception:
        if ws in _ws_clients: _ws_clients.remove(ws)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, timeout_keep_alive=120)
