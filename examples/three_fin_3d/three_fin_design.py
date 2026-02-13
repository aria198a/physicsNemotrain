# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
import os
import sys
import csv
import numpy as np

# --- [關鍵修正 1: 路徑修補，解決 ModuleNotFoundError] ---
core_parent = "/home/os-i-jingtai.chang/physicsnemo"
sym_parent = "/home/os-i-jingtai.chang/PhysicsNemo/physicsnemo-sym"
modulus_path = "/home/os-i-jingtai.chang/PhysicsNemo/modulus-sym"

sys.path = [p for p in sys.path if "physicsnemo" not in p.lower()]
sys.path.insert(0, modulus_path)
sys.path.insert(0, core_parent)
sys.path.insert(0, sym_parent)
# ------------------------------------------------------

from physicsnemo.sym.utils.io.csv_rw import dict_to_csv
from physicsnemo.sym.hydra import to_absolute_path

# --- [關鍵修正 2: 設定對齊你的 570 CFM 與訓練輸出] ---
max_pressure_drop = 2.5
num_design = 10

# 路徑必須指向 run_mode=eval 生成的新資料夾
path_flow = to_absolute_path("outputs/run_mode=eval/three_fin_flow")
path_thermal = to_absolute_path("outputs/run_mode=eval/three_fin_thermal")

# 參數名稱必須與 three_fin_geometry.py 完全一致
invar_mapping = [
    "fin_height_m",
    "fin_height_s",
    "fin_length_m",
    "fin_length_s",
    "fin_thickness_m",
    "fin_thickness_s",
]
outvar_mapping = ["pressure_drop", "peak_temp_K"]

def DesignOpt(path_flow, path_thermal, num_design, max_pressure_drop, invar_mapping, outvar_mapping):
    flow_monitors = os.path.join(path_flow, "monitors")
    thermal_monitors = os.path.join(path_thermal, "monitors")
    
    if not os.path.exists(flow_monitors):
        print(f"錯誤：找不到路徑 {flow_monitors}。請確認是否已跑過 eval 模式。")
        return

    values, configs = [], []

    # 搜尋評估產生的 CSV 文件
    for file in os.listdir(flow_monitors):
        if file.startswith("back_pressure") and file.endswith(".csv"):
            config_str = file[len("back_pressure"):-4]
            configs.append(config_str)

            value = []
            # 1. 讀取後壓
            with open(os.path.join(flow_monitors, file), "r") as f:
                reader = csv.reader(f)
                next(reader)
                value.append(float(list(reader)[-1][1]))

            # 2. 讀取前壓
            with open(os.path.join(flow_monitors, "front_pressure" + config_str + ".csv"), "r") as f:
                reader = csv.reader(f)
                next(reader)
                value.append(float(list(reader)[-1][1]))

            # 3. 讀取峰值溫度
            thermal_csv = os.path.join(thermal_monitors, "peak_temp" + config_str + ".csv")
            if os.path.exists(thermal_csv):
                with open(thermal_csv, "r") as f:
                    reader = csv.reader(f)
                    next(reader)
                    value.append(float(list(reader)[-1][1]))
            else:
                value.append(np.nan)
            
            values.append(value)

    if not values:
        print("未偵測到任何有效的監控數據。")
        return

    # 物理量轉換：還原開氏溫度
    data = np.array(values)
    pressure_drops = data[:, 1] - data[:, 0]
    peak_temps = (data[:, 2] + 1.0) * 273.15 
    
    # 篩選符合 570 CFM 壓力降限制的設計
    valid_idx = np.where(pressure_drops < max_pressure_drop)[0]
    if len(valid_idx) == 0:
        print(f"警告：沒有任何設計低於壓力降門檻 {max_pressure_drop}。")
        return
        
    f_values = np.column_stack((pressure_drops[valid_idx], peak_temps[valid_idx]))
    f_configs = [configs[i] for i in valid_idx]
    
    # 排序並選取最優設計
    best_idx = f_values[:, 1].argsort()[:num_design]
    opt_results = f_values[best_idx]
    opt_configs_str = [f_configs[i] for i in best_idx]

    # 生成最終報表
    opt_configs_data = np.array([[float(x) for x in c.split("_") if x] for c in opt_configs_str])
    final_dict = {
        **{k: opt_configs_data[:, i].reshape(-1, 1) for i, k in enumerate(invar_mapping)},
        **{k: opt_results[:, i].reshape(-1, 1) for i, k in enumerate(outvar_mapping)}
    }
    
    dict_to_csv(final_dict, "optimal_design_results")
    print("✅ 優化完成！結果已儲存至 optimal_design_results.csv")

if __name__ == "__main__":
    DesignOpt(path_flow, path_thermal, num_design, max_pressure_drop, invar_mapping, outvar_mapping)