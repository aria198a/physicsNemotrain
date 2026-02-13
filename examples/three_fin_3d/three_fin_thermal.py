# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
import os
import sys
import warnings
import torch
from sympy import Eq, tanh, Or, And
import itertools
import numpy as np
import wandb
from omegaconf import OmegaConf

# --- [強化修正：解決 ModuleNotFoundError: physicsnemo.sym.monitor] ---
core_parent = "/home/os-i-jingtai.chang/physicsnemo"
sym_parent = "/home/os-i-jingtai.chang/PhysicsNemo/physicsnemo-sym"
modulus_path = "/home/os-i-jingtai.chang/PhysicsNemo/modulus-sym"

# 1. 取得目前腳本目錄
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 清理舊路徑並按正確順序插入
sys.path = [p for p in sys.path if "physicsnemo" not in p.lower()]
sys.path.insert(0, modulus_path)
sys.path.insert(0, core_parent)
sys.path.insert(0, sym_parent)
sys.path.insert(0, script_dir)

# 3. 強制擴展 package 路徑，確保 sym 目錄下的模組能被找到
import physicsnemo
if hasattr(physicsnemo, "__path__"):
    physicsnemo.__path__.append(os.path.join(sym_parent, "physicsnemo"))
# ----------------------------------------------------------------------

from physicsnemo.sym.hydra.config import PhysicsNeMoConfig
from physicsnemo.sym.hydra import to_absolute_path
from physicsnemo.sym.utils.io import csv_to_dict
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.domain.monitor import PointwiseMonitor # 關鍵修復
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pdes.basic import GradNormal
from physicsnemo.sym.eq.pdes.diffusion import Diffusion, DiffusionInterface
from physicsnemo.sym.eq.pdes.advection_diffusion import AdvectionDiffusion

# 導入幾何定義
from three_fin_geometry import *

@physicsnemo.sym.main(config_path="conf", config_name="conf_thermal")
def run(cfg: PhysicsNeMoConfig) -> None:
    # --- [WandB 初始化：與流場專案連動] ---
    conf_dict = OmegaConf.to_container(cfg, resolve=True)
    clean_conf = {k: v for k, v in conf_dict.items() if v is not None and v != "???"}
    wandb.init(project="PhysicsNeMo-FPGA", name="Three-Fin-Thermal-Final", config=clean_conf)

    # 1. 物理方程式 (113.28W 熱源對應)
    ad = AdvectionDiffusion(T="theta_f", rho=1.0, D=0.02, dim=3, time=False)
    dif = Diffusion(T="theta_s", D=0.0625, dim=3, time=False)
    dif_inteface = DiffusionInterface("theta_f", "theta_s", 1.0, 5.0, dim=3, time=False)
    f_grad = GradNormal("theta_f", dim=3, time=False)
    s_grad = GradNormal("theta_s", dim=3, time=False)

    # 2. 網路架構 (載入流場權重，鎖定不訓練)
    input_keys = [Key("x"), Key("y"), Key("z")]
    if cfg.custom.parameterized:
        input_keys += [Key("fin_height_m"), Key("fin_height_s"), Key("fin_length_m"), 
                      Key("fin_length_s"), Key("fin_thickness_m"), Key("fin_thickness_s")]
    
    flow_net = FullyConnectedArch(input_keys=input_keys, output_keys=[Key("u"), Key("v"), Key("w"), Key("p")])
    thermal_f_net = FullyConnectedArch(input_keys=input_keys, output_keys=[Key("theta_f")])
    thermal_s_net = FullyConnectedArch(input_keys=input_keys, output_keys=[Key("theta_s")])

    thermal_nodes = (
        ad.make_nodes() + dif.make_nodes() + dif_inteface.make_nodes() +
        f_grad.make_nodes() + s_grad.make_nodes() +
        [flow_net.make_node(name="flow_network", optimize=False)] +
        [thermal_f_net.make_node(name="thermal_f_network")] +
        [thermal_s_net.make_node(name="thermal_s_network")]
    )

    geo = ThreeFin(parameterized=cfg.custom.parameterized)
    inlet_t = 293.15 / 273.15 - 1.0
    grad_t = 360 / 273.15 # 溫度梯度
    thermal_domain = Domain()

    # 3. 邊界條件設定
    thermal_domain.add_constraint(PointwiseBoundaryConstraint(
        nodes=thermal_nodes, geometry=geo.inlet, outvar={"theta_f": inlet_t},
        batch_size=cfg.batch_size.Inlet, criteria=Eq(x, channel_origin[0])), "inlet")

    # 4. 熱源定義 (使用 tanh 函數空間定位)
    sharpen_tanh = 60.0
    source_mask = (tanh(sharpen_tanh * (x - source_origin[0])) + 1.0) / 2.0 * \
                  (tanh(sharpen_tanh * ((source_origin[0] + source_dim[0]) - x)) + 1.0) / 2.0 * \
                  (tanh(sharpen_tanh * (z - source_origin[2])) + 1.0) / 2.0 * \
                  (tanh(sharpen_tanh * ((source_origin[2] + source_dim[2]) - z)) + 1.0) / 2.0
    
    thermal_domain.add_constraint(PointwiseBoundaryConstraint(
        nodes=thermal_nodes, geometry=geo.three_fin,
        outvar={"normal_gradient_theta_s": grad_t * source_mask},
        batch_size=cfg.batch_size.HeatSource, criteria=Eq(y, source_origin[1])), "heat_source")

    # 5. 啟動求解器
    wandb.watch(thermal_s_net, log="all", log_freq=100)
    thermal_slv = Solver(cfg, thermal_domain)
    thermal_slv.solve()
    wandb.finish()

if __name__ == "__main__":
    run()