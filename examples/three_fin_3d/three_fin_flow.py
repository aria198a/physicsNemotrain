# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
import os
import sys
import warnings
import torch
import numpy as np
import itertools
import wandb
from sympy import Symbol, Eq, Abs, tanh, Or, And
from omegaconf import OmegaConf

# --- [關鍵路徑修正：解決 ModuleNotFoundError: physicsnemo.sym.monitor] ---
core_parent = "/home/os-i-jingtai.chang/physicsnemo"
sym_parent = "/home/os-i-jingtai.chang/PhysicsNemo/physicsnemo-sym"
modulus_path = "/home/os-i-jingtai.chang/PhysicsNemo/modulus-sym"

# 強制按正確順序插入，確保 sym 組件被優先讀取
sys.path = [p for p in sys.path if "physicsnemo" not in p.lower()]
sys.path.insert(0, modulus_path)
sys.path.insert(0, core_parent)
sys.path.insert(0, sym_parent)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import physicsnemo
physicsnemo.__path__.append(os.path.join(sym_parent, "physicsnemo"))
# --------------------------------------------------------------------------

from physicsnemo.sym.hydra import to_absolute_path, PhysicsNeMoConfig
from physicsnemo.sym.utils.io import csv_to_dict
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.domain.monitor import PointwiseMonitor # 現在能找到了
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.sym.eq.pdes.turbulence_zero_eq import ZeroEquation
from physicsnemo.sym.eq.pdes.basic import NormalDotVec
from physicsnemo.sym.models.fully_connected import FullyConnectedArch

# 導入幾何定義
from three_fin_geometry import *

@physicsnemo.sym.main(config_path="conf", config_name="conf_flow")
def run(cfg: PhysicsNeMoConfig) -> None:
    # --- [修正：手動啟動 WandB 以顯示 1.5M 訓練曲線] ---
    conf_dict = OmegaConf.to_container(cfg, resolve=True)
    clean_conf = {k: v for k, v in conf_dict.items() if v is not None and v != "???"}
    wandb.init(project="PhysicsNeMo-FPGA", name="Three-Fin-Flow-Run", config=clean_conf)

    # 1. 物理方程式 (Navier-Stokes)
    if cfg.custom.turbulent:
        ze = ZeroEquation(nu=0.002, dim=3, time=False, max_distance=0.5)
        ns = NavierStokes(nu=ze.equations["nu"], rho=1.0, dim=3, time=False)
        navier_stokes_nodes = ns.make_nodes() + ze.make_nodes()
    else:
        ns = NavierStokes(nu=0.01, rho=1.0, dim=3, time=False)
        navier_stokes_nodes = ns.make_nodes()
    normal_dot_vel = NormalDotVec()

    # 2. 建立網路架構 (符合你的 RTX 4500 Ada 效能)
    input_keys = [Key("x"), Key("y"), Key("z")]
    if cfg.custom.parameterized:
        input_keys += [Key("fin_height_m"), Key("fin_height_s"), Key("fin_length_m"), 
                      Key("fin_length_s"), Key("fin_thickness_m"), Key("fin_thickness_s")]
    
    flow_net = FullyConnectedArch(input_keys=input_keys, output_keys=[Key("u"), Key("v"), Key("w"), Key("p")])
    flow_nodes = navier_stokes_nodes + normal_dot_vel.make_nodes() + [flow_net.make_node(name="flow_network")]

    geo = ThreeFin(parameterized=cfg.custom.parameterized)
    inlet_vel = 1.0  # 對應你的 570 CFM 風量需求
    flow_domain = Domain()

    # 3. 邊界約束 (Inlet/Outlet/NoSlip)
    u_profile = inlet_vel * tanh((0.5 - Abs(y)) / 0.02) * tanh((0.5 - Abs(z)) / 0.02)
    flow_domain.add_constraint(PointwiseBoundaryConstraint(
        nodes=flow_nodes, geometry=geo.inlet, outvar={"u": u_profile, "v": 0, "w": 0},
        batch_size=cfg.batch_size.Inlet, criteria=Eq(x, channel_origin[0])), "inlet")

    flow_domain.add_constraint(PointwiseBoundaryConstraint(
        nodes=flow_nodes, geometry=geo.geo, outvar={"u": 0, "v": 0, "w": 0},
        batch_size=cfg.batch_size.NoSlip), "no_slip")

    # 4. 監控器 (Monitor：這是你剛才報錯的地方)
    invar_inlet_pressure = geo.integral_plane.sample_boundary(1024, parameterization={**fixed_param_ranges, **{x_pos: -2}})
    flow_domain.add_monitor(PointwiseMonitor(invar_inlet_pressure, output_names=["p"], 
                             metrics={"inlet_pressure": lambda var: torch.mean(var["p"])}, nodes=flow_nodes))

    # 5. 啟動求解器並監控 Loss
    wandb.watch(flow_net, log="all", log_freq=100)
    flow_slv = Solver(cfg, flow_domain)
    flow_slv.solve()
    wandb.finish()

if __name__ == "__main__":
    run()