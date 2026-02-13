#!/usr/bin/env python3
import os
import sys
import csv
import argparse
import numpy as np
import torch

# =========================
# PhysicsNeMo namespace FIX
# =========================
core_parent = "/home/os-i-jingtai.chang/physicsnemo"
sym_parent = "/home/os-i-jingtai.chang/PhysicsNemo/physicsnemo-sym"
modulus_path = "/home/os-i-jingtai.chang/PhysicsNemo/modulus-sym"

sys.path = [p for p in sys.path if "physicsnemo" not in p.lower()]
sys.path.insert(0, modulus_path)
sys.path.insert(0, core_parent)
sys.path.insert(0, sym_parent)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import physicsnemo
physicsnemo.__path__.append(os.path.join(sym_parent, "physicsnemo"))

try:
    import physicsnemo.models.layers  # noqa
    import physicsnemo.sym  # noqa
    print("--- [PhysicsNeMo FIX] import OK: physicsnemo.sym + physicsnemo.models.layers ---")
except Exception as e:
    print(f"[ERROR] PhysicsNeMo import failed: {e}")
    print(f"physicsnemo.__path__ = {physicsnemo.__path__}")
    raise

# =========================
# Your geometry
# =========================
from fpga_geometry import *  # noqa: F401,F403

from physicsnemo.sym.key import Key
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.models.fourier_net import FourierNetArch
from physicsnemo.sym.models.siren import SirenArch
from physicsnemo.sym.models.modified_fourier_net import ModifiedFourierNetArch
from physicsnemo.sym.models.dgm import DGMArch


def build_model(arch: str, device: torch.device):
    input_keys = [Key("x"), Key("y"), Key("z")]
    # heat output 通常就是溫度 (T)，但有些腳本叫 "theta" 或 "t"
    # 這裡先用最常見 "T"；若 mismatch 我教你 10 秒修正
    output_keys = [Key("T")]

    if arch == "FullyConnectedArch":
        net = FullyConnectedArch(input_keys=input_keys, output_keys=output_keys)
    elif arch == "FourierNetArch":
        net = FourierNetArch(input_keys=input_keys, output_keys=output_keys)
    elif arch == "SirenArch":
        net = SirenArch(
            input_keys=input_keys,
            output_keys=output_keys,
            normalization={"x": (-2.5, 2.5), "y": (-2.5, 2.5), "z": (-2.5, 2.5)},
        )
    elif arch == "ModifiedFourierNetArch":
        net = ModifiedFourierNetArch(input_keys=input_keys, output_keys=output_keys)
    elif arch == "DGMArch":
        net = DGMArch(input_keys=input_keys, output_keys=output_keys, layer_size=128)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    net = net.to(device)
    net.eval()
    return net


def load_state(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model" in ckpt:
            sd = ckpt["model"]
        else:
            sd = ckpt
    else:
        sd = ckpt
    model.load_state_dict(sd, strict=False)


def make_grid_points(nx, ny, nz):
    x0, y0, z0 = channel_origin  # noqa: F405
    dx, dy, dz = channel_dim     # noqa: F405

    xs = np.linspace(x0, x0 + dx, nx)
    ys = np.linspace(y0, y0 + dy, ny)
    zs = np.linspace(z0, z0 + dz, nz)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=1)
    return pts


def filter_domain_points(pts: np.ndarray, domain: str):
    """
    domain:
      - fluid: channel 內部 (geo.sdf_channel > 0)
      - solid: solid 區域 (geo.sdf_solid > 0)  ← 如果沒有這個函數，我們就用差分法
    """
    if domain == "fluid":
        sdf = geo.sdf({"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2]}, {})  # noqa: F405
        keep = sdf["sdf"] > 0
        return pts[keep]

    if domain == "solid":
        # 有些 fpga_geometry 會有 geo_solid 或另一個 sdf；先嘗試常見命名
        if hasattr(geo, "sdf_solid"):
            sdf = geo.sdf_solid({"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2]}, {})  # type: ignore
            keep = sdf["sdf"] > 0
            return pts[keep]

        # fallback: solid = box內點 - fluid內點
        sdf = geo.sdf({"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2]}, {})  # noqa: F405
        fluid = sdf["sdf"] > 0
        return pts[~fluid]

    raise ValueError("domain must be fluid or solid")


@torch.no_grad()
def predict_in_batches(model, xyz: np.ndarray, batch_size: int, device: torch.device):
    N = xyz.shape[0]
    out_acc = None

    for i in range(0, N, batch_size):
        chunk = xyz[i:i + batch_size]
        x = torch.from_numpy(chunk[:, 0:1]).float().to(device)
        y = torch.from_numpy(chunk[:, 1:2]).float().to(device)
        z = torch.from_numpy(chunk[:, 2:3]).float().to(device)

        out = model({"x": x, "y": y, "z": z})

        if out_acc is None:
            out_acc = {k: [] for k in out.keys()}
        for k, v in out.items():
            out_acc[k].append(v.detach().cpu().numpy())

    out_np = {k: np.concatenate(vs, axis=0).reshape(-1) for k, vs in out_acc.items()}
    return out_np


def save_csv(path: str, pts: np.ndarray, pred: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = ["x", "y", "z"] + list(pred.keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for i in range(len(pts)):
            w.writerow([pts[i, 0], pts[i, 1], pts[i, 2]] + [pred[k][i] for k in pred.keys()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", required=True,
                    choices=["FullyConnectedArch", "FourierNetArch", "SirenArch", "ModifiedFourierNetArch", "DGMArch"])
    ap.add_argument("--domain", required=True, choices=["fluid", "solid"])
    ap.add_argument("--nx", type=int, default=100)
    ap.add_argument("--ny", type=int, default=50)
    ap.add_argument("--nz", type=int, default=50)
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--out", default="outputs/thermal.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    model = build_model(args.arch, device)
    print("[INFO] loading checkpoint ...")
    load_state(model, args.ckpt)

    print("[INFO] sampling grid points ...")
    pts = make_grid_points(args.nx, args.ny, args.nz)
    before = pts.shape[0]
    pts = filter_domain_points(pts, args.domain)
    print(f"[INFO] filter_domain({args.domain}): {before} -> {pts.shape[0]}")

    print("[INFO] running inference ...")
    pred = predict_in_batches(model, pts, args.batch, device)

    print("[INFO] saving csv ...")
    save_csv(args.out, pts, pred)
    print(f"[DONE] saved to: {args.out}")
    print("ParaView: Open CSV -> TableToPoints(x,y,z) -> Coloring by T -> Apply")


if __name__ == "__main__":
    main()
