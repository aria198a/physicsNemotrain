# 🚀 PhysicsNeMo: 40KW FPGA 散熱數位孿生專案

本專案利用 **NVIDIA PhysicsNeMo (Modulus)** 實現高效能散熱片的流熱耦合模擬。目前已完成流場高精度訓練，並進入 **113.28W** 熱源注入與能量守恆驗證階段。

## 📋 專案架構與檔案說明 (What to do)

根據專案開發流程圖，本倉庫包含以下核心組件：

| 檔案名稱 | 階段 | 具體任務 |
| :--- | :--- | :--- |
| **`fpga_geometry.py`** | **幾何層 (Geometry)** | 負責 40KW STL 的工程簡化、計算域與邊界標籤（Inlet/Outlet）定義。 |
| **`fpga_flow.py`** | **物理層 (Flow)** | 求解 Navier-Stokes 方程。目前已完成 **62 萬步** 訓練，Loss 達 $10^{-8}$。 |
| **`fpga_heat.py`** | **物理層 (Heat)** | 負責 **113.28W** 熱源注入。求解 Energy Eq. 並與凍結的流場權重進行耦合。 |
| **`conf_heat/`** | **配置層 (Config)** | 儲存訓練所需的 YAML 設定，包含權重讀取路徑與批次大小（Batch Size）。 |

---

## 🛠️ 執行與維護指南



### 1. 環境修復 (Environment Setup)
代碼中已整合路徑修復腳本，會自動搜尋並連結 `modulus-sym` 與 `physicsnemo-sym`。

### 2. 熱模擬啟動 (Heat Simulation)
目前熱模擬進度：**372,500 步**，Loss 穩定在 $3 \times 10^{-5}$。
* 執行指令：`python3 fpga_heat.py`
* 權重儲存路徑：`outputs/fpga_heat/network_checkpoint_heat`

### 3. 硬體故障排除 (GPU Driver)
若啟動時遇到 `CUDA error 999` 或 `NVML mismatch`：
* 請先執行：`sudo nvidia-smi -pm 1` (開啟 Persistence Mode)。
* 若無效，請確認 RTX 4500 Ada 驅動版本為 **580.126.09**。

---

## 🔍 物理層定義備忘 (Physics Details)

* **熱源採樣**：為了確保能捕捉到 STL 表面，邊界條件使用 `Abs(y - source_origin[1]) < 1e-3` 容差。
* **能量守恆**：使用了 `GradNormal` 節點來計算 `normal_gradient_theta_s`，確保符合 113.28W 的熱通量定義。

---

## 📊 成果視覺化
* **ParaView**：讀取 `vtp` 檔案，利用 **Glyph** 濾鏡觀察 570 CFM 風場。
* **Isaac Sim**：可透過 UDP 接收溫度數據，實時更新散熱片數位孿生模型。
