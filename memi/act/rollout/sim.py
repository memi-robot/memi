"""The RoboMME test environment with the `waypoint` action space, as the base harness builds it, and the wait action
this repo adds."""
from __future__ import annotations

import numpy as np

from .. import config, episodes


def make_env(task: str, episode: int):
    from robomme.env_record_wrapper import BenchmarkEnvBuilder
    return BenchmarkEnvBuilder(env_id=task, dataset=config.DATASET, action_space="waypoint", gui_render=False,
                               max_steps=config.MAX_STEPS).make_env_for_episode(
        episode, include_front_camera_extrinsic=True, include_front_camera_intrinsic=True,
        include_wrist_camera_extrinsic=True, include_wrist_camera_intrinsic=True)


def uint8(frame) -> np.ndarray:
    import torch
    return (frame.cpu().numpy() if isinstance(frame, torch.Tensor) else np.asarray(frame)).astype(np.uint8)


def png(frame) -> bytes:
    return episodes.png(uint8(frame))


def elapsed_steps(info: dict) -> int:
    return int(np.asarray(info.get("elapsed_steps", 0)).reshape(-1)[-1])


def calibration(obs: dict, info: dict) -> dict:
    return episodes.calibration(info["front_camera_intrinsic"], obs["front_camera_extrinsic_list"][-1],
                                info["wrist_camera_intrinsic"])


def robot_state(obs: dict, info: dict, start: int) -> dict:
    """The robot state as training shows it, with the control steps counted from `start`."""
    return episodes.robot_state(np.asarray(obs["eef_state_list"][-1]).reshape(-1),
                                np.asarray(obs["gripper_state_list"][-1]).reshape(-1), elapsed_steps(info) - start,
                                obs["wrist_camera_extrinsic_list"][-1])


def wait(env, k: int):
    """Hold still for k control steps, as the motion planner holds while it opens or closes the gripper. Each step
    repeats the joint targets read at the start and the gripper command the planner last gave (none on the stick
    tasks). Returns what the waypoint wrapper returns for a motion, and stops early when the episode ends."""
    from robomme.robomme_env.utils import planner_denseStep as dense
    ms = env.env                                  # FailAwareWrapper -> MultiStepDemonstrationWrapper
    try:
        planner = ms._get_planner()
        q = planner.robot.get_qpos()[0, : len(planner.planner.joint_vel_limits)].cpu().numpy()
        parts = [q, q * 0] if getattr(planner, "control_mode", "") == "pd_joint_pos_vel" else [q]
        if getattr(planner, "gripper_state", None) is not None:
            parts.append(np.asarray([planner.gripper_state], dtype=float))
        action = np.hstack(parts)
        steps = []
        for _ in range(k):
            steps += dense._output_to_steps(planner.env.step(action))
            if bool(dense._to_scalar(steps[-1][2])) or bool(dense._to_scalar(steps[-1][3])):
                break
        obs, rew, term, trunc, info = dense.to_step_batch(steps)
        return (obs, ms._take_last_step_value(rew), ms._take_last_step_value(term),
                ms._take_last_step_value(trunc), ms._flatten_info_batch(info))
    except Exception as e:                        # as FailAwareWrapper reports a failed step
        return None, 0.0, True, False, {"status": "error", "error_message": f"wait failed: {e!r}"}
