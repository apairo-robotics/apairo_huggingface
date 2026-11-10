"""Every number of hub/fomo_labels_card.md, regenerated from the data.

Reads the label tables (``--labels``: ``robosense/``, ``<camera>/``,
``preview/``), the per-trajectory state files written by
``fomo_labels_pipeline.py`` (mask checks, preview image quality) and the
``apairo_fomo_report.json`` of each trajectory (ATE). A trajectory the
pipeline has not finished prints ``TODO``.

    python scripts/fomo_card_numbers.py
"""

import argparse
import json
from pathlib import Path

import numpy as np

CAMERAS = ("zedx_left", "zedx_right", "basler")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/mnt/vault-fellowship/fomo"))
    p.add_argument("--logs", type=Path, default=Path("/mnt/vault-fellowship/fomo_logs"))
    p.add_argument("--labels", type=Path, default=Path("/mnt/vault-fellowship/fomo_labels"))
    args = p.parse_args()
    import pyarrow.parquet as pq

    trajs = sorted(f"{t.parent.name}/{t.name}" for t in args.root.glob("*/red_*")
                   if (t / "apairo_fomo_report.json").exists())

    # robosense: scans and driven shares, from the tables
    print("| Trajectory | Scans | future driven % | past driven % | scans re-checked |\n|---|---|---|---|---|")
    tot = np.zeros(4)
    for traj in trajs:
        dep, name = traj.split("/")
        f = args.labels / "robosense" / dep / f"{name}.parquet"
        st = json.loads((args.logs / "labels_pipeline" / f"{dep}__{name}.json").read_text()) \
            if (args.logs / "labels_pipeline" / f"{dep}__{name}.json").exists() else {}
        if not f.exists():
            print(f"| `{traj}` | TODO | TODO | TODO | TODO |")
            continue
        t = pq.read_table(f, columns=["n_points", "driven_indices", "driven_indices_past"])
        pts = int(np.sum(t.column("n_points").to_numpy()))
        fut = sum(len(x) for x in t.column("driven_indices").to_pylist())
        past = sum(len(x) for x in t.column("driven_indices_past").to_pylist())
        tot += [t.num_rows, pts, fut, past]
        print(f"| `{traj}` | {t.num_rows:,} | {100 * fut / pts:.2f} | {100 * past / pts:.2f} | "
              f"{st.get('scans_checked', 'TODO')} |")
    if tot[1]:
        print(f"| **rebuilt so far** | {int(tot[0]):,} | {100 * tot[2] / tot[1]:.2f} | {100 * tot[3] / tot[1]:.2f} | |")

    # cameras: frames, shares, bytes
    print("\n| Trajectory | Camera | Footprint | Listed frames | Masks | % 1 / 0 / 255 | Table MB |\n|---|---|---|---|---|---|---|")
    sizes = {c: 0 for c in ("robosense", *CAMERAS, "preview")}
    for traj in trajs:
        dep, name = traj.split("/")
        sp = args.logs / "labels_pipeline" / f"{dep}__{name}.json"
        st = json.loads(sp.read_text()) if sp.exists() else {}
        if st.get("step") not in ("labelled", "done"):
            print(f"| `{traj}` | all | | TODO | TODO | TODO | TODO |")
            continue
        for c in CAMERAS:
            m = st["masks"][c]
            sh = " / ".join(f"{100 * m['share'][k]:.1f}" for k in ("1", "0", "255"))
            print(f"| `{traj}` | {c} | {m['footprint']} | {m['listed_frames']:,} | {m['masks']:,} | {sh} | "
                  f"{st['table_bytes'][c] / 1e6:.0f} |")
        for c in ("robosense", *CAMERAS):
            sizes[c] += st["table_bytes"][c]
        sizes["preview"] += st["preview_bytes"]
    print("\ntable sizes over the rebuilt trajectories (MB): "
          + ", ".join(f"{k} {v / 1e6:.0f}" for k, v in sizes.items()))

    print("\npreview image quality (mean intensity / saturated share, R, G, B >= 250):")
    for traj in trajs:
        dep, name = traj.split("/")
        sp = args.logs / "labels_pipeline" / f"{dep}__{name}.json"
        st = json.loads(sp.read_text()) if sp.exists() else {}
        q = st.get("preview_quality")
        if not q:
            print(f"  {traj}: TODO")
            continue
        print(f"  {traj}: " + "; ".join(
            f"{c} {q[c]['mean_intensity']:.1f} / {100 * q[c]['saturated_share']:.2f} % ({q[c]['frames']} frames)"
            for c in CAMERAS))

    ate = [json.loads((args.root / t / "apairo_fomo_report.json").read_text())["ate_rmse_m"] for t in trajs]
    print(f"\nKISS-ICP ATE RMSE over {len(ate)} trajectories: {min(ate):.2f}-{max(ate):.2f} m")


if __name__ == "__main__":
    main()
