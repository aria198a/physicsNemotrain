🚀 PhysicsNeMo: 40KW FPGA 散熱數位孿生專案

本專案利用 NVIDIA PhysicsNeMo (Modulus) 實現高效能散熱片的流熱耦合模擬。目前已完成流場高精度訓練，並進入 113.28W 熱源注入階段。
📋 檔案清單與功能說明
檔案名稱	負責項目	關鍵技術 / 參數
fpga_geometry.py	幾何層 (Geometry)	定義 40KW STL 幾何、Inlet/Outlet 區域、以及計算域。
fpga_flow.py	流體層 (Physics - Flow)	執行 Navier-Stokes 模擬。目前已達 621,900 步，Loss 10−8。
fpga_heat.py	熱學層 (Physics - Heat)	注入 113.28W 熱源。結合 Energy Eq. 與 Advection-Diffusion 方程。
conf_heat/config.yaml	配置層 (Config)	定義訓練參數、Batch Size、以及權重載入路徑。
🛠️ 核心操作流程 (Workflow)

    幾何簡化：透過 fpga_geometry.py 將真實 CAD 轉換為 STL 並進行工程簡化。

    流場預訓練：執行 fpga_flow.py。確保流場收斂後，權重會存於 network_checkpoint_flow。

    熱耦合模擬：

        讀取已凍結的流場權重（Optimize=False）。

        執行 fpga_heat.py 注入熱源功率。

        目前熱模擬已突破 37 萬步，Loss 穩定在 10−5 等級。
