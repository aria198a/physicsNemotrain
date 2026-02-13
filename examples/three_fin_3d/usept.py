import torch
import os
import sys

# --- [路徑修正] ---
sys.path.insert(0, "/home/os-i-jingtai.chang/physicsnemo")
sys.path.insert(0, "/home/os-i-jingtai.chang/PhysicsNemo/physicsnemo-sym")
sys.path.insert(0, "/home/os-i-jingtai.chang/PhysicsNemo/modulus-sym")

from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.key import Key

# 1. 定義架構 (必須匹配你的 9 維輸入與 1 維輸出)
input_keys = [
    Key("x"), Key("y"), Key("z"),
    Key("fin_height_m"), Key("fin_height_s"),
    Key("fin_length_m"), Key("fin_length_s"),
    Key("fin_thickness_m"), Key("fin_thickness_s")
]
output_keys = [Key("theta_s")]

model = FullyConnectedArch(input_keys=input_keys, output_keys=output_keys)

# 2. 載入權重檔
ckpt_path = "outputs/three_fin_thermal/thermal_s_network.0.pth"
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    model.eval()
    print(f"成功載入模型: {ckpt_path}")
else:
    print(f"找不到檔案: {ckpt_path}")
    sys.exit()

# 3. 準備輸入數據 (x, y, z 以及 6 個尺寸參數)
# 這裡輸入一組特定座標與你訓練時的基礎尺寸 (0.4, 1.0, 0.1)
invar = {
    "x": torch.tensor([[0.0]]), 
    "y": torch.tensor([[0.0]]), 
    "z": torch.tensor([[0.0]]),
    "fin_height_m": torch.tensor([[0.4]]),
    "fin_height_s": torch.tensor([[0.4]]),
    "fin_length_m": torch.tensor([[1.0]]),
    "fin_length_s": torch.tensor([[1.0]]),
    "fin_thickness_m": torch.tensor([[0.1]]),
    "fin_thickness_s": torch.tensor([[0.1]]),
}

# 4. 進行預測 (像 YOLO 偵測一樣快！)
with torch.no_grad():
    prediction = model(invar)
    # 溫度還原：(normalized_val + 1) * 273.15
    temp_k = (prediction["theta_s"].item() + 1.0) * 273.15
    print(f"座標 (0,0,0) 的預測溫度為: {temp_k:.2f} K")