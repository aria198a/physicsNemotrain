#!/usr/bin/env python3
import numpy as np
import torch
import matplotlib.pyplot as plt
from fpga_geometry import solid

# 1. 建立隨機採樣點（覆蓋整個 Channel 範圍）
N = 20000 
x = np.random.uniform(-2.5, 2.5, N)
y = np.random.uniform(-0.5, 0.5, N)
z = np.zeros(N) # 固定在中間切面

# 2. 準備輸入
invar = {
    "x": torch.tensor(x[:, None], dtype=torch.float32),
    "y": torch.tensor(y[:, None], dtype=torch.float32),
    "z": torch.tensor(z[:, None], dtype=torch.float32)
}

# 3. 執行 SDF 計算
print(f"[INFO] Sampling {N} points for STL visualization...")
sdf_results = solid.sdf(invar, {})
sdf = sdf_results["sdf"]
if hasattr(sdf, "detach"): sdf = sdf.detach().cpu().numpy()

# 4. 強制對齊長度 (如果長度不一，取最小集)
min_len = min(len(x), len(sdf))
x_plot = x[:min_len]
y_plot = y[:min_len]
sdf_plot = sdf[:min_len].flatten()

# 5. 繪圖：只畫出內部點 (SDF > 0)
plt.figure(figsize=(12, 5))
mask = sdf_plot > 0
plt.scatter(x_plot[mask], y_plot[mask], c='red', s=1, label='Inside STL')
plt.scatter(x_plot[~mask], y_plot[~mask], c='blue', s=1, alpha=0.1, label='Outside')

plt.xlim(-2.5, 2.5)
plt.ylim(-0.5, 0.5)
plt.title("40KW STL Geometry Mapping (Red = Detected Solid)")
plt.legend()
plt.savefig("sdf_scatter.png")
print("✅ Visualization saved as 'sdf_scatter.png'.")