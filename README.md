# 🚀 PhysicsNeMo: 三鰭片散熱器設計空間優化 (PINNs)

本專案利用 **NVIDIA PhysicsNeMo (Modulus)** 實現參數化散熱片的流熱耦合模擬。
核心目標是在 **570 CFM** 風量限制下，優化 **113.28W** 熱源的散熱效能，並實現毫秒級的物理預測。

## 📋 專案開發流程與組件

根據 PINNs 開發流程，本專案分為以下四個核心階段：

| 檔案名稱 | 階段 | 技術亮點 |
| :--- | :--- | :--- |
| **`three_fin_geometry.py`** | **幾何層** | 定義參數化三鰭片，支援鰭片高度、長度、厚度的動態調整。 |
| **`three_fin_flow.py`** | **流場訓練** | 求解 Navier-Stokes 方程，Loss 達到 **$10^{-8}$** 的極高精度。 |
| **`three_fin_thermal.py`** | **熱傳訓練** | 注入 **113.28W** 熱源，Loss 達到 **$10^{-10}$**，精確模擬熱對流與傳導。 |
| **`three_fin_design.py`** | **設計優化** | 自動在數百組幾何組合中搜尋「壓損最小」且「溫度最低」的最佳設計。 |

---

## 🔍 關鍵技術實作

### 1. 即時推論 (Real-time Inference)
不同於傳統 CFD 需要數小時計算，本專案訓練出的 `.pth` 模型可實現「像 YOLO 一樣快」的推論：
* **單點預測**：使用 `usept.py` 可瞬間獲取特定座標溫度（例如原點預測值：**446.33 K**）。
* **大規模場預測**：`inference_three_fin.py` 可生成 25 萬點的 `.csv` 數據供 ParaView 視覺化。

### 2. 高精度權重管理
模型權重存放於 `outputs/` 目錄，確保訓練過程可追蹤：
* **流場權重**：`flow_network.0.pth`
* **熱傳權重**：`thermal_s_network.0.pth` (固體) 與 `thermal_f_network.0.pth` (流體)

---

## 🛠️ 快速啟動指南

### 環境修復
腳本內建路徑鎖定邏輯，會自動連結 `physicsnemo-sym` 與核心庫。

### 視覺化驗證
1. **ParaView**：讀取 `flow_results.csv` 或 `thermal_results.csv`。
2. **Table To Points**：將座標轉為 3D 點雲觀察溫度梯度與流線。

---

## 📊 專案成果備忘
- **輸入風量**: 570 CFM (5.0 m/s 基準)
- **熱源功率**: 113.28 W (GradNormal 熱通量定義)
- **優化目標**: 壓力降 < 2.5 Pa
