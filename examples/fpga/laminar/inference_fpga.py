#!/usr/bin/env python3
import os
import sys

# =========================
# PhysicsNeMo import fix (TEMP PATCH)
# =========================
def _physicsnemo_fix_paths():
    """
    目的：讓 python 能同時找到
      - physicsnemo.models (core)
      - physicsnemo.sym    (sym)
    避免 pip editable build isolation 問題時先救急跑 inference。
    """
    sym_parent   = "/home/os-i-jingtai.chang/PhysicsNemo/physicsnemo-sym"
    core_parent  = "/home/os-i-jingtai.chang/physicsnemo"              # ⚠️你要確定這裡底下真的有 physicsnemo/models
    modulus_path = "/home/os-i-jingtai.chang/PhysicsNemo/modulus-sym"  # 可有可無

    def safe_prepend(p):
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    # 優先：目前腳本位置 → sym → core → modulus
    safe_prepend(os.path.dirname(os.path.abspath(__file__)))
    safe_prepend(sym_parent)
    safe_prepend(core_parent)
    safe_prepend(modulus_path)

    # 清掉已載入的 physicsnemo，避免改 sys.path 後仍用舊模組
    for m in list(sys.modules.keys()):
        if m == "physicsnemo" or m.startswith("physicsnemo."):
            del sys.modules[m]

    import physicsnemo

    # 把 sym 的 package 位置加入 namespace
    sym_pkg = os.path.join(sym_parent, "physicsnemo")
    if os.path.isdir(sym_pkg) and sym_pkg not in list(physicsnemo.__path__):
        physicsnemo.__path__.append(sym_pkg)

    # 把 core 的 package 位置加入 namespace（確保 models 找得到）
    core_pkg = os.path.join(core_parent, "physicsnemo")
    if os.path.isdir(core_pkg) and core_pkg not in list(physicsnemo.__path__):
        physicsnemo.__path__.append(core_pkg)

    # 診斷輸出（跑一次就知道到底吃到哪裡）
    print("=== [PhysicsNeMo FIX] sys.executable ===")
    print(sys.executable)
    print("=== [PhysicsNeMo FIX] physicsnemo.__path__ ===")
    print(list(physicsnemo.__path__))

    # 真正測試
    import physicsnemo.sym
    import physicsnemo.models.layers
    print("--- [PhysicsNeMo FIX] import OK: physicsnemo.sym + physicsnemo.models.layers ---")

_physicsnemo_fix_paths()

# =========================
# normal imports
# =========================
import csv
import argparse
import numpy as np
import torch

from fpga_geometry import *  # noqa: F401,F403

from physicsnemo.sym.key import Key
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.models.fourier_net import FourierNetArch
from physicsnemo.sym.models.siren import SirenArch
from physicsnemo.sym.models.modified_fourier_net import ModifiedFourierNetArch
from physicsnemo.sym.models.dgm import DGMArch


def build_model(arch: str, exact_continuity: bool, device: torch.device):
    # 與訓練腳本一致
    input_keys = [Key("x"), Key("y"), Key("z")]
    if exact_continuity:
        output_keys = [Key("a"), Key("b"), Key("c"), Key("p")]
    else:
        output_keys = [Key("u"), Key("v"), Key("w"), Key("p")]

    if arch == "FullyConnectedArch":
        net = FullyConnectedArch(input_keys=input_keys, output_keys=output_keys)
    elif arch == "FourierNetArch":
        net = FourierNetArch(input_keys=input_keys, output_keys=output_keys)
    elif arch == "SirenArch":
        # Siren 訓練時有 normalization，推論要一樣
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


@torch.no_grad()
def predict_in_batches(model, xyz: np.ndarray, batch_size: int, device: torch.device):
    """
    xyz: (N,3) numpy
    return: dict of numpy arrays
    """
    N = xyz.shape[0]
    out_acc = None

    for i in range(0, N, batch_size):
        chunk = xyz[i:i + batch_size]
        x = torch.from_numpy(chunk[:, 0:1]).float().to(device)
        y = torch.from_numpy(chunk[:, 1:2]).float().to(device)
        z = torch.from_numpy(chunk[:, 2:3]).float().to(device)

        invar = {"x": x, "y": y, "z": z}
        out = model(invar)  # PhysicsNeMo Sym arch 直接吃 dict，吐 dict

        # 第一次初始化 accumulator
        if out_acc is None:
            out_acc = {k: [] for k in out.keys()}

        for k, v in out.items():
            out_acc[k].append(v.detach().cpu().numpy())

    # concat
    out_np = {k: np.concatenate(vs, axis=0).reshape(-1) for k, vs in out_acc.items()}
    return out_np


def make_grid_points(nx, ny, nz):
    """
    建一個規則網格，範圍用 channel_origin / channel_dim
    注意：這個只是「盒狀 grid」；如果你要更精確的幾何內點，下面有 filter。
    """
    x0, y0, z0 = channel_origin  # noqa: F405
    dx, dy, dz = channel_dim     # noqa: F405

    xs = np.linspace(x0, x0 + dx, nx)
    ys = np.linspace(y0, y0 + dy, ny)
    zs = np.linspace(z0, z0 + dz, nz)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=1)
    return pts


def filter_inside_geometry(pts: np.ndarray):
    """
    用 geo.sdf 只保留幾何內部點（sdf > 0 通常代表 inside）
    你訓練時也在用 sdf 做 criteria。
    """
    # geo 來自 fpga_geometry.py
    sdf = geo.sdf({"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2]}, {})  # noqa: F405
    inside = sdf["sdf"] > 0
    return pts[inside]


def save_csv(path: str, pts: np.ndarray, pred: dict):
    """
    輸出 ParaView 友善 CSV：x,y,z + fields
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    field_names = ["x", "y", "z"] + list(pred.keys())

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(field_names)
        for i in range(pts.shape[0]):
            row = [pts[i, 0], pts[i, 1], pts[i, 2]] + [pred[k][i] for k in pred.keys()]
            writer.writerow(row)


def load_state(model, ckpt_path):
    """
    盡量兼容幾種常見存法：
    1) torch.save(model.state_dict())
    2) torch.save({"model": state_dict, ...})
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model" in ckpt:
            sd = ckpt["model"]
        else:
            # 可能整包就是 state_dict
            sd = ckpt
    else:
        sd = ckpt

    model.load_state_dict(sd, strict=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="path to flow_network.0.pth")
    ap.add_argument("--arch", type=str, required=True,
                    choices=["FullyConnectedArch", "FourierNetArch", "SirenArch", "ModifiedFourierNetArch", "DGMArch"])
    ap.add_argument("--exact_continuity", action="store_true",
                    help="if training used exact_continuity (outputs a,b,c,p)")
    ap.add_argument("--nx", type=int, default=80)
    ap.add_argument("--ny", type=int, default=40)
    ap.add_argument("--nz", type=int, default=40)
    ap.add_argument("--batch", type=int, default=65536, help="inference batch size")
    ap.add_argument("--no_filter_inside", action="store_true",
                    help="do not filter points by geo.sdf (use full box grid)")
    ap.add_argument("--out", type=str, default="outputs/flow_pred.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    model = build_model(args.arch, args.exact_continuity, device)
    print("[INFO] loading checkpoint ...")
    load_state(model, args.ckpt)
    model.eval()

    print("[INFO] sampling grid points ...")
    pts = make_grid_points(args.nx, args.ny, args.nz)

    if not args.no_filter_inside:
        before = pts.shape[0]
        pts = filter_inside_geometry(pts)
        print(f"[INFO] filter_inside: {before} -> {pts.shape[0]} points")

    print("[INFO] running inference ...")
    pred = predict_in_batches(model, pts, args.batch, device)

    print("[INFO] saving csv ...")
    save_csv(args.out, pts, pred)
    print(f"[DONE] saved to: {args.out}")
    print("ParaView: File -> Open -> flow_pred.csv; then 'Table To Points' (x,y,z).")


if __name__ == "__main__":
    main()
