"""KISS-ICP drift against FoMo's GNSS-PPK ground truth (``gt.txt``).

``gt.txt`` gives positions only (its orientations are identity), so the score
is the absolute trajectory error on positions after a rigid (Umeyama, no
scale) alignment. The GNSS stamps are the reference clock; the poses are
SE(3)-interpolated at each of them by ``synchronize()``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from apairo_transform.interp import Se3Interp

from . import config
from .labeling import load_poses, open_trajectory


def ate(traj_dir) -> dict:
    traj_dir = Path(traj_dir)
    gt = np.loadtxt(traj_dir / "gt.txt", ndmin=2)
    ds = open_trajectory(traj_dir, [config.POSES])
    _, ts = load_poses(ds)
    gt = gt[(gt[:, 0] > ts[0]) & (gt[:, 0] < ts[-1])]
    sync = ds.synchronize(
        reference=gt[:, 0], method={config.POSES: Se3Interp()}, tolerance=config.POSE_TOLERANCE
    )
    stamps = np.array([sync[k].timestamp for k in range(len(sync))])
    est = np.stack([np.asarray(sync[k].data[config.POSES])[:3, 3] for k in range(len(sync))])
    ref = gt[np.searchsorted(gt[:, 0], stamps), 1:4]
    mu_e, mu_r = est.mean(0), ref.mean(0)
    U, _, Vt = np.linalg.svd((ref - mu_r).T @ (est - mu_e))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    err = np.linalg.norm(est @ R.T + (mu_r - R @ mu_e) - ref, axis=1)
    length = float(np.linalg.norm(np.diff(ref, axis=0), axis=1).sum())
    rmse = float(np.sqrt((err ** 2).mean()))
    return {"frames": len(err), "path_m": length, "ate_rmse_m": rmse, "ate_max_m": float(err.max())}
