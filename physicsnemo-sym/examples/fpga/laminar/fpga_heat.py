import sys
import os

# 1. 環境路徑徹底修復
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
    import physicsnemo.sym
    print("--- [PhysicsNeMo] 環境修復完成，準備啟動 GPU 熱模擬 ---")
except Exception as e:
    print(f"導入失敗: {e}")

# 2. 導入幾何定義
from fpga_geometry import *

# 確保 fpga 與 solid 變數名稱對齊
if 'solid' in locals() and 'fpga' not in locals():
    fpga = solid
    print("✅ 已將 solid 幾何映射為 fpga 變數")

import torch
from sympy import Eq, tanh, Abs  # 使用大寫 Abs 解決 SymPy 導入問題
import numpy as np
from physicsnemo.sym.hydra import PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import PointwiseBoundaryConstraint, PointwiseInteriorConstraint
from physicsnemo.sym.domain.monitor import PointwiseMonitor
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pdes.basic import GradNormal # 修正：導入導數節點
from physicsnemo.sym.eq.pdes.diffusion import Diffusion, DiffusionInterface
from physicsnemo.sym.eq.pdes.advection_diffusion import AdvectionDiffusion
from physicsnemo.sym.models.fully_connected import FullyConnectedArch

@physicsnemo.sym.main(config_path="conf_heat", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # --- [修正] 定義熱源座標 (物理層核心) ---
    source_origin = [0.0, 0.0, 0.0]  # 請確認這與你 STL 底部位置一致
    source_dim = [0.1, 0.0, 0.1]     # 113.28W 發熱區域
    
    # 物理參數 (113.28W 專案)
    rho = 1
    k_fluid = 1.0
    k_solid = 5.0
    D_solid = 0.10
    D_fluid = 0.02
    source_grad = 1.5 

    # 建立 PDE 節點
    ad = AdvectionDiffusion(T="theta_f", rho=rho, D=D_fluid, dim=3, time=False)
    dif = Diffusion(T="theta_s", D=D_solid, dim=3, time=False)
    dif_inteface = DiffusionInterface("theta_f", "theta_s", k_fluid, k_solid, dim=3, time=False)
    
    # --- [核心修復] 補上法向梯度節點，解決 Unroll Graph 失敗 ---
    f_grad = GradNormal("theta_f", dim=3, time=False)
    s_grad = GradNormal("theta_s", dim=3, time=False) 

    input_keys = [Key("x"), Key("y"), Key("z")]
    output_keys = [Key("u"), Key("v"), Key("w"), Key("p")]

    # 載入你訓練 62 萬步的 10^-8 精度流場權重
    flow_net = FullyConnectedArch(input_keys=input_keys, output_keys=output_keys)
    thermal_f_net = FullyConnectedArch(input_keys=input_keys, output_keys=[Key("theta_f")])
    thermal_s_net = FullyConnectedArch(input_keys=input_keys, output_keys=[Key("theta_s")])

    thermal_nodes = (
        ad.make_nodes() + dif.make_nodes() + dif_inteface.make_nodes() +
        f_grad.make_nodes() + s_grad.make_nodes() + # 加入導數計算節點
        [flow_net.make_node(name="flow_network", optimize=False)] + 
        [thermal_f_net.make_node(name="thermal_f_network")] +
        [thermal_s_net.make_node(name="thermal_s_network")]
    )

    thermal_domain = Domain()

    # --- 113.28W 熱源注入 ---
    sharpen_tanh = 60.0
    source_func = (
        (tanh(sharpen_tanh * (x - source_origin[0])) + 1.0) / 2.0 *
        (tanh(sharpen_tanh * ((source_origin[0] + source_dim[0]) - x)) + 1.0) / 2.0 *
        (tanh(sharpen_tanh * (z - source_origin[2])) + 1.0) / 2.0 *
        (tanh(sharpen_tanh * ((source_origin[2] + source_dim[2]) - z)) + 1.0) / 2.0
    )
    
    thermal_domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=thermal_nodes, 
            geometry=fpga, 
            outvar={"normal_gradient_theta_s": source_grad * source_func},
            batch_size=cfg.batch_size.heat_source, 
            # 使用 Abs 與容差解決 "no surface" 報錯
            criteria=Abs(y - source_origin[1]) < 1e-3 
        ), "heat_source")

    # 內域能量守恆損失
    thermal_domain.add_constraint(
        PointwiseInteriorConstraint(
            nodes=thermal_nodes, geometry=geo, outvar={"advection_diffusion_theta_f": 0},
            batch_size=cfg.batch_size.hr_flow_interior
        ), "hr_flow_interior")

    # 啟動 Solver
    thermal_slv = Solver(cfg, thermal_domain)
    thermal_slv.solve()

if __name__ == "__main__":
    run()