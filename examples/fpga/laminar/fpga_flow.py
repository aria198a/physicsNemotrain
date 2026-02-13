import sys
import os
import importlib.util

# =========================================================
# 🛠️ 第一部分：環境路徑與大小寫 (P big N big) 強制修復
# =========================================================
modulus_path = "/home/os-i-jingtai.chang/PhysicsNemo/modulus-sym"
core_parent = "/home/os-i-jingtai.chang/PhysicsNemo/physicsnemo-sym" 

# 清理並注入正確路徑
sys.path = [p for p in sys.path if "physicsnemo" not in p.lower()]
sys.path.insert(0, modulus_path)
sys.path.insert(0, core_parent)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import physicsnemo
physicsnemo.__path__ = [os.path.join(core_parent, "physicsnemo")]

# 修正 FCLayer 導入路徑
try:
    from physicsnemo.nn import FCLayer, get_activation
    print("✅ [PhysicsNeMo] FCLayer 導入成功 (來自 nn 模組)")
except ImportError:
    from physicsnemo.models.layers import FCLayer
    print("✅ [PhysicsNeMo] FCLayer 導入成功 (來自 models 模組)")

# =========================================================
# 📦 第二部分：物理模擬組件導入
# =========================================================
import torch
import warnings
from sympy import Symbol, Eq, And, Or
import numpy as np

from physicsnemo.sym.hydra import to_absolute_path, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.domain.monitor import PointwiseMonitor
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.sym.eq.pdes.basic import NormalDotVec
from physicsnemo.sym.models.fully_connected import FullyConnectedArch

# 導入台達 40KW STL 幾何定義
from fpga_geometry import *

# =========================================================
# 🌊 第三部分：求解器主程式 (對接台達風扇規格)
# =========================================================
@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # --- 1. 物理參數 (台達 PFB 570 CFM 換算) ---
    nu = 0.02           # 運動黏度
    rho = 1.0           # 空氣密度
    inlet_vel = 0.223   # 入口風速 (0.223 m/s)
    
    # 建立 sympy 符號
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    x_pos = Symbol("x_pos")

    # 【修復】明確定義採樣方塊 (確保 40KW 零件被包圍)
    # 若 fpga_geometry.py 內無定義，則在此建立
    global flow_box_origin, flow_box_dim, integral_plane
    if 'flow_box_origin' not in globals():
        flow_box_origin = [channel_origin[0] + 0.1, channel_origin[1], channel_origin[2]]
        flow_box_dim = [1.0, channel_dim[1], channel_dim[2]]
        
    if 'integral_plane' not in globals():
        # 若無定義，則使用入口後方 0.1m 的平面作為監控點
        from physicsnemo.sym.geometry.primitives_3d import Plane
        integral_plane = Plane(
            (channel_origin[0] + 0.1, channel_origin[1], channel_origin[2]),
            (channel_origin[0] + 0.1, channel_origin[1] + channel_dim[1], channel_origin[2] + channel_dim[2])
        )

    # --- 2. 建立 PDE 節點 ---
    ns = NavierStokes(nu=nu, rho=rho, dim=3, time=False)
    normal_dot_vel = NormalDotVec()
    
    flow_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        adaptive_activations=cfg.custom.adaptive_activations,
    )
    flow_nodes = ns.make_nodes() + normal_dot_vel.make_nodes() + [flow_net.make_node(name="flow_network")]

    # --- 3. 建立 Domain 與 約束條件 ---
    flow_domain = Domain()

    # 入口 (Inlet)
    def channel_sdf(x, y, z):
        return channel.sdf({"x": x, "y": y, "z": z}, {})["sdf"]

    flow_domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=flow_nodes,
            geometry=inlet,
            outvar={"u": inlet_vel, "v": 0, "w": 0},
            batch_size=cfg.batch_size.inlet,
            criteria=Eq(x, channel_origin[0]),
            lambda_weighting={"u": channel_sdf, "v": 1.0, "w": 1.0},
        ), "inlet"
    )

    # 出口 (Outlet)
    flow_domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=flow_nodes,
            geometry=outlet,
            outvar={"p": 0},
            batch_size=cfg.batch_size.outlet,
            criteria=Eq(x, channel_origin[0] + channel_dim[0]),
        ), "outlet"
    )

    # 40KW STL 表面 (No-Slip)
    flow_domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=flow_nodes,
            geometry=geo,
            outvar={"u": 0, "v": 0, "w": 0},
            batch_size=cfg.batch_size.no_slip,
        ), "no_slip"
    )

    # 內部高解析度採樣 (針對 40KW 散熱片細節)
    flow_domain.add_constraint(
        PointwiseInteriorConstraint(
            nodes=flow_nodes,
            geometry=geo,
            outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
            batch_size=cfg.batch_size.hr_interior,
            criteria=And(x > flow_box_origin[0], x < (flow_box_origin[0] + flow_box_dim[0])),
        ), "hr_interior"
    )

    # --- 4. 監控器 (壓力降監測) ---
    invar_front_p = integral_plane.sample_boundary(
        1024, parameterization={x_pos: channel_origin[0] + 0.1}
    )
    flow_domain.add_monitor(
        PointwiseMonitor(
            invar_front_p,
            output_names=["p"],
            metrics={"front_pressure": lambda var: torch.mean(var["p"])},
            nodes=flow_nodes,
        )
    )

    # --- 5. Solver 執行 ---
    flow_slv = Solver(cfg, flow_domain)
    flow_slv.solve()

if __name__ == "__main__":
    run()