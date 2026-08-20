# Phase 2 handoff: PX4/Gazebo/ROS2 deployment of the trained MARL policy

> **ACTIVE WORK IN PROGRESS (branch `px4-deployment-wip`, 2026-08-20).**
> This branch is a live PX4/Gazebo/ROS2 deployment debugging session (Phase
> 7 observation-adapter verification, N=4 checkpoint evaluation, ongoing).
> It was branched off `origin/vision-tracking` after the shared worktree at
> `.claude/worktrees/vision-tracking` got switched to `xyz-spread-fixed` by
> another concurrent session mid-task, which would have kept clobbering
> this work. If you're working on reward-shaping / `xyz-spread` / the
> N-aware-margin line of work, please don't force-push over this branch or
> merge into it without checking in first -- ping before touching
> `deployment/` here. Not blocking any other branch; just flagging so we
> don't step on each other.

Written 2026-08-20 at the end of the first live deployment session, to let
a fresh conversation resume without re-deriving anything below. Read
`TESTING_METHODOLOGY.md` first -- it's the protocol this effort now
follows. This file is the concrete state: what's built, what's confirmed
working, what broke, and exactly where to pick up.

**Project context**: Stage_Marl's end goal is edge AI -- the trained
policy eventually runs on a real drone's onboard compute, not just
simulation. This PX4/Gazebo/ROS2 pipeline is the architecture that carries
over to real hardware (same node pattern: telemetry in, trained-policy
inference, action out to the flight controller), so it's core project
work, not a demo.

## Phase 0 -- frozen environment (verified 2026-08-20)

| Fact | Value |
|---|---|
| OS | Ubuntu 24.04.4 LTS (noble) |
| ROS 2 | Jazzy |
| Gazebo | Gazebo Sim 8.14.0 (Harmonic-era) |
| PX4 commit | `7bf22ca4a0` on `/home/rayen/PX4-Autopilot` |
| Python | 3.12.3 (`/usr/bin/python3`) |
| PyTorch | 2.13.0+cu130 |
| RMW implementation | unset / ROS 2 default |
| Repo working dir | `/home/rayen/stage/.claude/worktrees/vision-tracking` (a git worktree, branch `vision-tracking` -- NOT the main checkout) |
| Model checkpoint | `models/actor_best_n3_3_20260817-1255_b59c139-3m-seed3.pt` (N=3 agents) |
| Relevant ROS 2 packages present | `px4_msgs`, `px4_ros_com`, `ros_gz`, `ros_gz_bridge`, `ros_gz_image`, `ros_gz_interfaces`, `ros_gz_sim`, `ros_gz_sim_demos` |

**Sourcing required before anything ROS2/px4_msgs-related** (both lines,
in this order -- the second one is what actually provides `px4_msgs`, and
its absence fails with `ModuleNotFoundError: No module named 'px4_msgs'`,
not an obvious ROS2-config error):

```bash
source /opt/ros/jazzy/setup.bash
source /home/rayen/ws_sensor_combined/install/setup.bash
```

**`gz` PATH gotcha**: the ROS2-vendored `gz` that ends up on `PATH` by
default is broken for this workflow. Always prefix commands with
`PATH="/usr/bin:$PATH"` or call `/usr/bin/gz` explicitly.

**Env vars required to run `deployment/inference_node.py`**:
`NUM_AGENTS=3` (config.py defaults to 4 -- `NUM_AGENTS=int(os.environ.get("NUM_AGENTS", 4))`
-- and the script hard-asserts `NUM_AGENTS==3` on import, but you still
have to export it yourself before launch or the assert fails).

**PX4 SITL paths**:
`/home/rayen/PX4-Autopilot/build/px4_sitl_default/bin/px4` (binary, also
multicall-symlinked as `px4-param`, `px4-commander`, `px4-listener`,
`px4-ekf2`, etc.), per-instance state at
`/home/rayen/PX4-Autopilot/build/px4_sitl_default/rootfs/<instance>/` (this
is also each instance's required cwd when running the `px4-*` client
tools against it, e.g. `cd rootfs/0 && px4-commander check`). World:
`/home/rayen/PX4-Autopilot/Tools/simulation/gz/worlds/default.sdf`.
Vehicle model: `gz_x500` (`PX4_SIM_MODEL=gz_x500`,
`PX4_SYS_AUTOSTART=4001`). DDS bridge: one shared
`MicroXRCEAgent udp4 -p 8888` serves all 3 SITL instances.

This is a single-agent test scaled to 3 real agents (no phantoms) -- see
"What's built" below.

## What's built

- **`deployment/inference_node.py`** -- the ROS2 node. One process per
  real drone (`--agent-name drone1|drone2|drone3`). Each instance
  subscribes to *all three* agents' real telemetry (own + both
  neighbours) on PX4's namespaced topics, drives a reused
  `envs.formation_env.FormationEnv3D` instance's internal state by hand
  every cycle from that real telemetry plus a scripted target, calls
  `_get_obs()` for the exact training observation, runs the actor,
  applies `FormationEnv3D._apply_brake` (the same closing-speed safety
  layer used in training, extracted as its own method specifically so
  this node can reuse it verbatim), and publishes
  OffboardControlMode/TrajectorySetpoint/VehicleCommand. Namespace map:
  `drone1=""` (PX4 instance 0), `drone2="/px4_1"` (instance 1),
  `drone3="/px4_2"` (instance 2).
- **`deployment/launch_px4_instance.sh <instance> <x> <y>`** -- launches
  one PX4 SITL instance. Instance 0 starts the Gazebo world; instances 1
  and 2 attach standalone (`PX4_GZ_STANDALONE=1`) at a given spawn pose.
  All three now correctly set `PX4_GZ_MODEL_POSE` (see bug #3 below).
- **`deployment/sim_demo.py`** -- pure-Python sanity check of the trained
  checkpoint against the *abstract* `FormationEnv3D` (no PX4/Gazebo at
  all). This is effectively Phase 1+2 of the methodology and is already
  done: a full 200-step deterministic episode against this exact
  checkpoint completed cleanly, no collision, no target_lost.
- **Spawn geometry**: `TARGET_START=(0,0,3)` (sim/Gazebo frame, z-up).
  `SPAWN_XY = {drone1: (5.0, 0.0), drone2: (-2.5, 4.33), drone3: (-2.5, -4.33)}`
  -- a radius-5 circle around the target at exact 120° spacing, i.e. the
  scene starts already close to the trained-for equilateral formation.
- **Target**: a single scripted, visible Gazebo marker (not a 4th real
  vehicle) -- deliberate, not a shortcut: the vision-tracking system
  already models "sometimes visible, sensor-range-limited" detection
  (`_update_target_track`), so a scripted target exercises that logic
  honestly. Owned/driven by drone1's process once drone1 finishes
  takeoff; drifts at `PHANTOM_SPEED=0.3 m/s` in `+Y`. The other two
  processes compute the identical deterministic trajectory independently
  (no shared Python state between the 3 OS processes) rather than reading
  it from Gazebo.
- **Explicit takeoff phase**: each node climbs straight up to
  `TAKEOFF_ALTITUDE=3.0` under a hand-coded constant-velocity climb before
  handing off to the trained policy -- the policy itself has no notion of
  "take off from the ground," tracking is its whole job.
- **Coordinate handling**: PX4 telemetry/setpoints are NED (Z-down); sim
  frame is Z-up, single sign-flip conversion (`ned_to_sim`/`sim_to_ned`).
  Each PX4 instance's `vehicle_local_position` is relative to *its own*
  local EKF origin near its own spawn point, not a frame shared across
  instances -- `SPAWN_OFFSET` (== `SPAWN_XY`) is added back to every
  position read (own and both neighbours') to reconstruct one shared
  world frame.

## Four confirmed, fixed bugs (this session, 2026-08-20)

All four were root-caused with actual evidence (PX4 source + live
telemetry), per Rule 2 -- not guessed. Do not re-litigate these; they are
verified.

1. **GPS Horizontal Position Drift preflight check false-tripping.** All
   3 PX4 instances got stuck forever at "Arming denied: Resolve system
   health failures first" / "Preflight: GPS Horizontal Pos Drift too
   high" (`estimator_status.gps_check_fail_flags` bit 5, threshold
   `EKF2_REQ_HDRIFT=0.1 m/s`). Confirmed via live telemetry that the
   EKF's own fused position accuracy was ~1.5cm (not actually drifting) --
   this is the raw-GPS-sample drift check false-tripping, plausibly from
   uneven message timing under this machine's 3-instance load. Fix:
   `EKF2_GPS_CHECK` bitmask 2047 → 2015 (clears only bit 5,
   "Horizontal position drift" -- the other 10 GPS quality/safety checks
   stay enabled). Must be re-applied per PX4 instance after every fresh
   launch (`cd rootfs/<i> && px4-param set EKF2_GPS_CHECK 2015`), it does
   not persist across a fresh daemon start.
2. **One-shot arm attempt, silent forever on failure.**
   `inference_node.py`'s `_timer_callback` used to call `_engage_offboard()`
   + `_arm()` exactly once (at `setpoint_counter==10`) and never again,
   with zero logging in the "not offboard or not armed" fallback branch --
   a rejected arm attempt meant the process sat there forever producing no
   output at all, indistinguishable from "still working." Fixed: retries
   every 2s while not armed+offboard, and logs the wait state
   (`nav_state=`, `armed=`) on the same cadence. This is a real,
   permanent fix, not just a diagnostic aid -- keep it.
3. **drone1 (PX4 instance 0) spawn-offset mismatch.**
   `launch_px4_instance.sh`'s instance-0 branch never set
   `PX4_GZ_MODEL_POSE`, so it always spawned at Gazebo's world default
   `(0,0)` instead of the code's assumed `(5.0, 0.0)`. Confirmed via
   PX4's own `px4-rc.gzsim` startup script source that
   `PX4_GZ_MODEL_POSE` is honored identically regardless of whether the
   instance is creating the world or attaching standalone -- there was no
   reason for the asymmetry. This silently corrupted every position
   calculation for drone1 by a constant 5m in X: its own belief about its
   distance to the target, *and* what drone2/drone3 perceived as drone1's
   position (since neighbour positions are reconstructed via
   `SPAWN_OFFSET` too). Fixed by adding the missing export. Confirmed via
   Gazebo `pose/info`: drone1 now spawns at exactly `(5, 0)`.
4. **Stale EKF references after 2+ days of continuous PX4 daemon
   uptime.** Even after fixing #1 and #3, all 3 instances still failed
   preflight with a *different* check: `pre_flt_fail_innov_pos_horiz`
   ("horizontal position unstable"), `pos_test_ratio` pegged at ~1.8-2.0
   (gate is 1.0) and flat, not converging, for over a minute. Diagnosed
   via raw `estimator_innovations.gps_hpos`: **~20-30 meters** of
   disagreement between GPS-reported and EKF-believed position -- a real,
   structural mismatch, not noise. Restarting just the `ekf2` module
   in-place did *not* fix it (ruled out "stale overconfident covariance"
   as the mechanism). Root cause: the 3 PX4 daemons had been running
   continuously for 2+ days through multiple earlier Gazebo
   teardown/respawn cycles in this session without themselves ever being
   restarted, so each EKF's internal position reference had gone stale
   relative to the vehicle's actual current world pose. Fix: full stack
   restart (kill all 3 PX4 daemons + Gazebo, relaunch everything fresh).
   Confirmed via `gps_hpos` innovations dropping to the millimeter/
   centimeter range immediately after. **Practical implication: always do
   a full PX4+Gazebo restart when resuming this work, not an in-place
   respawn of Gazebo models against already-running PX4 daemons.**

## Current state: everything killed, clean slate

All PX4 daemons, Gazebo (GUI + server), the 3 inference_node.py
processes, and the shared MicroXRCEAgent were killed at the end of this
session. Nothing is running. Start fresh from Phase 0/3 using the recipe
above rather than trying to inherit any prior state.

## Open problem -- this is where to resume

With all 4 bugs above fixed, a full clean run was done: all 3 real drones
armed, took off cleanly, and handed off to the trained policy. For the
first ~15-20 seconds, tracking looked genuinely good -- `dist_to_target`
for drone1 and drone2 converged toward and hovered near the ideal
(TARGET_DIST ≈ 4.5-4.78 depending on N). Then, over the following
~20-40 seconds, it degraded:

- drone1 drifted onto an increasingly diverging trajectory (sustained
  travel in `+Y`, climbing in `Z`), `dist_to_target` climbing back past 7,
  and eventually crashed hard enough to flip (~180° roll, confirmed via
  Gazebo `pose/info` orientation quaternion) -- PX4 logged
  `Attitude failure (roll)`, which then blocks re-arming entirely.
- drone2 and drone3 both sank to near-ground level (`z≈0`) and got
  `Disarmed by landing` by PX4's land detector, then stayed grounded
  (the retry-arm logic from bug fix #2 kept re-arming them, but nothing
  in the code re-triggers the takeoff climb once `self.took_off` is
  already `True`, so they just sat there running the trained policy from
  a grounded state).
- `min_dist_to_neighbor` grew to 12-19m during the divergence -- the
  drones were separating from each other, not colliding.

Applying the methodology's own failure template:

```text
FAILURE:
All 3 real drones track the target well for ~15-20s after handoff to the
trained policy, then diverge -- two ground themselves, one crashes hard
enough to flip.

LAST VERIFIED STAGE:
Phase 6 equivalent (deterministic multi-instance PX4/Gazebo/ROS2 mechanics,
arming, takeoff, coordinate-frame reconstruction) -- confirmed working via
the 4 bug fixes above. Closed-loop 3-agent flight (Phase 12-15 territory)
was attempted directly after that. Phases 7-11 (observation adapter
comparison, observation sanity tests, observation distribution comparison,
action adapter validation, shadow mode) were skipped entirely.

FIRST UNVERIFIED INTERFACE:
The observation adapter and action adapter have never been validated in
isolation. inference_node.py's obs construction (env.pos/vel/pos_t driven
by hand from real telemetry, then env._get_obs()) has not been compared
element-by-element against what training/sim_demo.py produces for an
equivalent state. The action → safe_vel_sim → NED setpoint conversion has
not been sanity-tested with synthetic actions.

EXPECTED:
dist_to_target converges toward TARGET_DIST and stays there once the
policy takes over, indefinitely (matching sim_demo.py's clean 200-step
run).

ACTUAL:
Converges for ~15-20s (~300-400 policy steps at 20Hz), then diverges.

EVIDENCE:
Raw inference_node.py log lines from the last run tonight (pos=,
dist_to_target=, min_dist_to_neighbor=, safe_vel_sim=, brake=,
real/fallback_cycles=); PX4 daemon logs ("Disarmed by landing" x3,
"Attitude failure (roll)" for drone1); Gazebo pose/info orientation
quaternion confirming drone1's flip. No structured CSV/plot exists yet --
this evidence is all from scrollback, which is itself a gap (see Phase 18
below).

MOST LIKELY CAUSE:
(1) Sim-to-real gap: FormationEnv3D is a pure kinematic model (direct
position integration, no physics, no ground plane, no crash consequence)
-- a policy that's unpenalized for dipping toward low altitude in the
abstract sim has no reason to have learned that a real ground strike is
catastrophic, especially if the target's z and/or the agents' natural
formation geometry brings them near low altitude at all.
(2) Actual achieved control-loop rate under this machine's real load (3x
PX4 SITL + 3x torch inference + Gazebo GUI+server is heavy) may not
actually be holding DT=0.05/20Hz -- not yet measured. A policy trained at
a fixed control interval can behave very differently if the real interval
drifts under load, and degradation-after-tens-of-seconds is consistent
with a load-dependent effect kicking in once the whole stack is busier
(3 real policies inferring + brake computation + neighbour bookkeeping,
vs. sim_demo.py's single-agent single-process test).
(3) All 3 agents starting simultaneously from a dead stop at exactly
TARGET_DIST from the target may be out-of-distribution relative to
training's reset() distribution -- worth checking what reset() actually
randomizes vs. this fixed, synchronized start.

DO NOT CHANGE:
The 4 already-confirmed-and-fixed bugs above (GPS check, arm retry, drone1
spawn offset, full-restart-not-in-place-respawn). These are verified via
direct evidence, not places to keep digging.

NEXT SINGLE TEST:
Do not attempt another live 3-agent flight yet. Go back and do the skipped
phases in order:
  (a) Phase 7 -- instrument inference_node.py (or a small standalone
      script reusing its exact obs-construction path) to dump the
      constructed observation vector, labeled by index/name, for one real
      drone during real flight; diff it element-by-element against
      FormationEnv3D._get_obs() on an equivalent synthetic state fed the
      same positions/velocities.
  (b) Phase 17 -- measure actual wall-clock time between consecutive
      _timer_callback invocations during a real 3-agent run, to see
      whether the loop is actually holding 20Hz under this machine's real
      load or silently degrading.
  (c) Phase 9 -- once (a) is instrumented, let one drone fly for the same
      ~20-40s window that previously preceded divergence, and compare the
      resulting observation distribution (especially altitude/z-relative
      and velocity components) against whatever range the training
      environment actually produces for a converged formation.
Whichever of (a)/(b)/(c) turns up a concrete discrepancy first is the
actual bug; don't touch the trained network itself until ROS→PX4→Gazebo
and the observation/action contracts are proven clean per Rule 3.
```

## Secondary, lower-priority gap noted along the way

`inference_node.py`'s `self.took_off` flag is set once and never reset --
if a drone touches the ground mid-flight for any reason, it will never
attempt to re-climb; it just keeps running the trained policy from a
grounded state forever. Worth deciding whether a re-takeoff path is
wanted once the actual root cause of the descent/crash is found -- low
priority until then, since fixing it would just mask the real problem
with a recovery loop.
