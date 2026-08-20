"""Verifies the two angular quantities r_spread and TARGET_DIST/_PACKING_RATIO
depend on -- added 2026-08-20 after a supervisor review caught that the
original xyz-spread implementation conflated them (used the global
target-viewpoint angle, 109.47/120 deg, where the local drone-viewpoint
angle, 60 deg, was needed). See envs/formation_env.py's
_IDEAL_NEIGHBOR_ANGLE comment for the narrative; this is the check that
would have caught the bug.

Two different angles, both real, not interchangeable:
  - GLOBAL (target's viewpoint): angle between two drones, as seen from
    the target/center. This is what config.py's _PACKING_RATIO and
    TARGET_DIST are built from.
  - LOCAL (a drone's viewpoint): angle between two neighbors, as seen from
    one drone. This is what _get_reward's r_spread penalizes deviation
    from, via _IDEAL_NEIGHBOR_ANGLE.
"""
import itertools
import math
import numpy as np

from envs.formation_env import _IDEAL_NEIGHBOR_ANGLE


def angle(u, v):
    return math.acos(float(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0, 1.0)))


def regular_simplex_vertices(k):
    """k+1 equidistant, pairwise-equidistant points (standard basis vectors,
    a textbook regular-simplex construction) -- k=2 gives an equilateral
    triangle (N=3 formation), k=3 gives a regular tetrahedron (N=4)."""
    return [np.eye(k + 1)[i] for i in range(k + 1)]


for k, label in [(2, "N=3 (triangle)"), (3, "N=4 (tetrahedron)")]:
    verts = regular_simplex_vertices(k)
    centroid = np.mean(verts, axis=0)
    centered = [v - centroid for v in verts]

    global_angles = {angle(centered[i], centered[j]) for i, j in itertools.combinations(range(len(verts)), 2)}
    assert len(global_angles) == 1, f"{label}: global angle not constant across pairs: {global_angles}"
    global_angle = next(iter(global_angles))

    a = centered[0]
    local_dirs = [centered[i] - a for i in range(1, len(centered))]
    local_angles = {round(angle(local_dirs[i], local_dirs[j]), 9)
                     for i, j in itertools.combinations(range(len(local_dirs)), 2)}
    assert len(local_angles) == 1, f"{label}: local angle not constant across pairs: {local_angles}"
    local_angle = next(iter(local_angles))

    print(f"{label}: global (target-viewpoint) = {math.degrees(global_angle):.2f} deg, "
          f"local (drone-viewpoint) = {math.degrees(local_angle):.2f} deg")

    assert abs(local_angle - math.pi / 3) < 1e-9, (
        f"{label}: local drone-viewpoint angle should be exactly 60 deg for any regular "
        f"simplex, got {math.degrees(local_angle):.4f} deg"
    )
    assert abs(local_angle - _IDEAL_NEIGHBOR_ANGLE) < 1e-9, (
        f"{label}: _IDEAL_NEIGHBOR_ANGLE ({math.degrees(_IDEAL_NEIGHBOR_ANGLE):.2f} deg) "
        f"doesn't match the analytically-verified local angle "
        f"({math.degrees(local_angle):.2f} deg) -- r_spread would be rewarding the wrong geometry."
    )

print("OK -- _IDEAL_NEIGHBOR_ANGLE matches the analytically-verified regular-simplex "
      "local angle for both N=3 and N=4.")
