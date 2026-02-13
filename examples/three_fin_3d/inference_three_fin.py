#!/usr/bin/env python3
import os
import sys
import csv
import argparse
import numpy as np
import torch

# =========================
# PhysicsNeMo 路徑修正 (與你之前的成功經驗一致)
# =========================
def _physicsnemo_fix_paths():
    sym_parent   = "/home/os-i-jingtai.chang/PhysicsNemo/physicsnemo-sym"
    core_parent  = "/home/os-i-jingtai.chang/physicsnemo"
    modulus_path = "/home/os-i-jingtai.chang/PhysicsNemo/modulus-sym"

    def safe_prepend(p):
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    safe_prepend(os.path.dirname(os.path.abspath(__file__)))
    safe_prepend(sym_parent)
    safe_prepend(core_parent)
    safe_prepend(modulus_path)

    import physicsnemo
    for pkg in [sym_parent, core_parent]:
        pkg_path = os.path.join(pkg, "physicsnemo")
        if os.path.isdir(pkg_path) and pkg_path not in list(physicsnemo.__path__):
            physicsnemo.__path__.append(pkg_path)

_physicsnemo_fix_paths()

# =========================
# 導入幾何與 PhysicsNeMo 組件
# =========================
from three_fin_geometry import ThreeFin, channel_origin, channel_dim 
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.fully_connected import FullyConnectedArch

def build_model(arch: str, device: torch.device, is_thermal: bool = False):
    # 訓練時使用了 9 個輸入 (3 座標 + 6 幾何參數)
    input_keys = [
        Key("x"), Key("y"), Key("z"),
        Key("fin_height_m"), Key("fin_height_s"),
        Key("fin_length_m"), Key("fin_length_s"),
        Key("fin_thickness_m"), Key("fin_thickness_s")
    ]
    
    if is_thermal:
        # --- [關鍵修正：熱傳權重 thermal_s...pth 為單一輸出 theta_s] ---
        output_keys = [Key("theta_s")] 
    else:
        # 流場權重 flow_network...pth 為四個輸出
        output_keys = [Key("u"), Key("v"), Key("w"), Key("p")]

    if arch == "FullyConnectedArch":
        net = FullyConnectedArch(input_keys=input_keys, output_keys=output_keys)
    else:
        raise ValueError(f"目前腳本僅支援 FullyConnectedArch")

    return net.to(device).eval()

@torch.no_grad()
def predict_in_batches(model, xyz: np.ndarray, batch_size: int, device: torch.device):
    N = xyz.shape[0]
    out_acc = None

    # 設定推論的特定幾何尺寸 (基準設計)
    h_m, h_s = 0.4, 0.4
    l_m, l_s = 1.0, 1.0
    t_m, t_s = 0.1, 0.1

    for i in range(0, N, batch_size):
        chunk = xyz[i:i + batch_size]
        n_chunk = chunk.shape[0]
        
        invar = {
            "x": torch.from_numpy(chunk[:, 0:1]).float().to(device),
            "y": torch.from_numpy(chunk[:, 1:2]).float().to(device),
            "z": torch.from_numpy(chunk[:, 2:3]).float().to(device),
            "fin_height_m": torch.full((n_chunk, 1), h_m).to(device),
            "fin_height_s": torch.full((n_chunk, 1), h_s).to(device),
            "fin_length_m": torch.full((n_chunk, 1), l_m).to(device),
            "fin_length_s": torch.full((n_chunk, 1), l_s).to(device),
            "fin_thickness_m": torch.full((n_chunk, 1), t_m).to(device),
            "fin_thickness_s": torch.full((n_chunk, 1), t_s).to(device),
        }
        out = model(invar)

        if out_acc is None:
            out_acc = {k: [] for k in out.keys()}
        for k, v in out.items():
            out_acc[k].append(v.detach().cpu().numpy())

    return {k: np.concatenate(vs, axis=0).reshape(-1) for k, vs in out_acc.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="pth 檔案路徑")
    ap.add_argument("--type", choices=["flow", "thermal"], default="flow")
    ap.add_argument("--nx", type=int, default=100)
    ap.add_argument("--ny", type=int, default=50)
    ap.add_argument("--nz", type=int, default=50)
    ap.add_argument("--out", type=str, default="results_pred.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    is_thermal = (args.type == "thermal")
    model = build_model("FullyConnectedArch", device, is_thermal)
    
    print(f"[INFO] 正在載入權重: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    # 核心修復：使用 strict=False 確保參數化權重能正確對應
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)

    x0, y0, z0 = channel_origin
    dx, dy, dz = channel_dim
    xs = np.linspace(x0, x0 + dx, args.nx)
    ys = np.linspace(y0, y0 + dy, args.ny)
    zs = np.linspace(z0, z0 + dz, args.nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=1)

    print(f"[INFO] 正在進行推論...")
    pred = predict_in_batches(model, pts, 65536, device)

    field_names = ["x", "y", "z"] + list(pred.keys())
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(field_names)
        for i in range(pts.shape[0]):
            row = [pts[i, 0], pts[i, 1], pts[i, 2]] + [pred[k][i] for k in pred.keys()]
            writer.writerow(row)
    
    print(f"[DONE] 結果已儲存至: {args.out}")

if __name__ == "__main__":
    main()