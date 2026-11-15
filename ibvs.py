"""
Image-based visual servoing (IBVS) towards a single target: Tasks 1 and 2.

Provided:
  - IBVSServo.run : the servo loop (measure the feature, compute the camera
                    velocity, map it to joint velocities, move the robot, log)

Student TODOs:
  - ibvs_camera_velocity        : TODO 2  classic IBVS control law          (Task 1)
  - estimate_target_motion      : TODO 3  feature rate caused by the target (Task 2)
  - compensated_camera_velocity : TODO 4  IBVS for a moving target          (Task 2)
"""

import numpy as np

from config import (ADAPTIVE_GAIN, CAMERA_OFFSET_POS, CONTROL_DT, IBVS_GAIN, JOINT_LIMITS_LOWER,
                    JOINT_LIMITS_UPPER, JOINT_VELOCITY_LIMITS, MOTION_FILTER, PREDICTION_BETA,
                    PREDICTION_STEPS, USE_ADAPTIVE_GAIN, USE_MOTION_PREDICTION)
from kinematics import InteractionMatrix, damped_pinv


# --------------------- Student TODOs -------------------------------------

def ibvs_camera_velocity(L, e, lam):
    """Classic IBVS: camera twist v (6,) such that the image error decays
    exponentially, e_dot = -lam * e.

    L   : 3x6 interaction matrix at the current feature
    e   : image error s - s* (3,)
    lam : gain (1/s)
    """
    # ------------------------------------------------------------------
    # TODO 2 (Task 1): classic IBVS control law
    #   v = -lam * L+ e          (use damped_pinv(L) for L+)
    # return v
    # ------------------------------------------------------------------
    L_pinv = damped_pinv(L)          # pseudo-inverse of the 3x6 matrix (6x3)
    v = -lam * L_pinv @ e            # camera twist [vx, vy, vz, wx, wy, wz]
    return v


def estimate_target_motion(s, s_prev, L_prev, v_prev, dt):
    """Feature rate caused by the motion of the target itself (3,).

    Between two measurements the feature changes because the camera moved
    (L_prev @ v_prev) and because the target moved (de/dt).  The measured
    rate is (s - s_prev) / dt, so the target's part is what remains.
    """
    # ------------------------------------------------------------------
    # TODO 3 (Task 2): estimate the target-induced feature rate
    #   de_hat = (s - s_prev) / dt - L_prev @ v_prev
    # return de_hat
    # ------------------------------------------------------------------
    measured_rate = (s - s_prev) / dt      # how fast the feature really changed
    camera_rate = L_prev @ v_prev          # the part caused by our own camera motion
    de_hat = measured_rate - camera_rate   # what is left was caused by the target
    return de_hat


def compensated_camera_velocity(L, e, lam, de_hat):
    """IBVS for a moving target: cancel the target's motion and reduce the error.

    The error obeys e_dot = L v + de/dt.  Choose v so that
    e_dot = -lam * e, using the estimate de_hat of de/dt.
    """
    # ------------------------------------------------------------------
    # TODO 4 (Task 2): compensated IBVS control law
    #   v = -L+ (lam * e + de_hat)
    # return v
    # ------------------------------------------------------------------
    L_pinv = damped_pinv(L)
    v = -L_pinv @ (lam * e + de_hat)       # reduce the error AND cancel the target motion
    return v


# --------------------- Improvements --------------------------------------

def adaptive_gain(err_norm, l0, linf, slope0):
    """Adaptive gain (the form used in ViSP): l0 when the error is zero,
    linf when it is large, and slope0 = d(lambda)/d|e| at zero error."""
    return (l0 - linf) * np.exp(-slope0 * err_norm / (l0 - linf)) + linf


# --------------------- Servo loop (provided) -----------------------------

class IBVSServo:
    """Servo the camera to the goal feature with IBVS.

    compensate=False uses the classic law (TODO 2); compensate=True adds the
    target-motion estimate (TODOs 3 and 4).
    """

    def __init__(self, env, kinematics, goal, lam=IBVS_GAIN, dt=CONTROL_DT, compensate=False):
        self.env = env
        self.kin = kinematics
        self.goal = np.asarray(goal, dtype=np.float64)
        self.lam = float(lam)
        self.dt = float(dt)
        self.compensate = compensate
        self.Lm = InteractionMatrix()

    def _cam_jac(self, q):
        return self.kin.compute_camera_jacobian(
            q, physics_client=self.env.physics_client,
            camera_offset_pos=getattr(self.env, "camera_offset_pos", CAMERA_OFFSET_POS),
            camera_mount_rotation=getattr(self.env, "_camera_mount_rotation", None))

    def run(self, steps, target_motion=None, step_callback=None):
        """Servo for ``steps`` control periods.

        target_motion(t) gives the target position at time t (None = static).
        Stops early if the target leaves the image or the robot touches an obstacle.
        """
        env = self.env
        q = np.asarray(env.q0, dtype=np.float64).copy()
        env.reset_joint_state(q)
        log = {k: [] for k in ("t", "features", "errors", "xy_errors", "camera_velocities",
                               "joint_velocities", "joints", "motion_estimates", "clearance")}
        status, collision_q = "completed", None
        s_prev = L_prev = v_prev = None
        de_hat = np.zeros(3)
        trend = np.zeros(3)        # how fast de_hat is changing (improvement 2)

        for k in range(int(steps)):
            t = k * self.dt
            if target_motion is not None:
                env.set_target_position(target_motion(t))
            s, z, rgb, mask = env.get_current_feature(return_mask=True)
            if s is None:
                status = "target_lost"
                break
            s = np.asarray(s, dtype=np.float64)
            L = self.Lm.compute(s, z)
            e = s - self.goal

            # Improvement 1: the gain rises as the error gets small
            lam = self.lam
            if USE_ADAPTIVE_GAIN:
                lam = adaptive_gain(np.linalg.norm(e), **ADAPTIVE_GAIN)

            if self.compensate:
                if s_prev is not None:
                    new = estimate_target_motion(s, s_prev, L_prev, v_prev, self.dt)
                    old = de_hat
                    de_hat = (1 - MOTION_FILTER) * de_hat + MOTION_FILTER * np.asarray(new)
                    if USE_MOTION_PREDICTION:
                        change = (de_hat - old) / self.dt
                        trend = (1 - PREDICTION_BETA) * trend + PREDICTION_BETA * change
                # Improvement 2: use the target motion predicted a few steps ahead,
                # because de_hat is built from old, smoothed measurements
                de_used = de_hat
                if USE_MOTION_PREDICTION:
                    de_used = de_hat + trend * PREDICTION_STEPS * self.dt
                v = compensated_camera_velocity(L, e, lam, de_used)
            else:
                v = ibvs_camera_velocity(L, e, lam)
            v = np.asarray(v, dtype=np.float64)

            # Camera twist -> joint velocities, within the joint velocity limits
            q_dot = damped_pinv(self._cam_jac(q)) @ v
            ratio = float(np.max(np.abs(q_dot) / JOINT_VELOCITY_LIMITS))
            q_dot = q_dot / ratio if ratio > 1 else q_dot
            q_next = np.clip(q + q_dot * self.dt, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)
            if env.obstacle_ids and env.check_collision(q_next, obstacle_margin=0.0):
                status, collision_q = "collision", q_next
                break

            log["t"].append(t)
            log["features"].append(s)
            log["errors"].append(float(np.linalg.norm(e)))
            log["xy_errors"].append(float(np.linalg.norm(e[:2])))
            log["camera_velocities"].append(v)
            log["joint_velocities"].append(q_dot)
            log["joints"].append(q.copy())
            log["motion_estimates"].append(de_hat.copy())
            if env.obstacle_ids:
                log["clearance"].append(env.min_obstacle_distance())
            if callable(step_callback):
                step_callback({"step": k + 1, "waypoint_idx": k, "total_waypoints": steps,
                               "target_waypoint": self.goal, "current_feature": s, "depth": z,
                               "tracking_error": float(np.linalg.norm(e[:2])),
                               "goal_error": float(np.linalg.norm(e)), "joint_positions": q,
                               "joint_velocities": q_dot, "rgb": rgb, "mask": mask,
                               "status": "Servoing...",
                               "clearance": log["clearance"][-1] if log["clearance"] else None})

            s_prev, L_prev, v_prev = s, L, v
            q = q_next
            env.reset_joint_state(q)                   # move the robot

        out = {k: np.asarray(v) for k, v in log.items()}
        out["status"] = status
        out["final_q"] = q
        out["collision_q"] = collision_q
        print(f"[IBVS] {status} after {len(out['t'])} steps; "
              f"final error {out['errors'][-1] if len(out['errors']) else float('nan'):.1f} px")
        return out
