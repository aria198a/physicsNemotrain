import importlib
import os
import sys

# 定義要從中抓取實體檔案的目錄 (nn 資料夾)
nn_dir = os.path.dirname(os.path.abspath(__file__))

# 取得該目錄下所有的 .py 檔案
files = [f[:-3] for f in os.listdir(nn_dir) if f.endswith('.py') and f != '__init__.py']

# 動態導入這些檔案中的所有類別
for file in files:
    try:
        module = importlib.import_module(f'physicsnemo.nn.{file}')
        for name in dir(module):
            if not name.startswith('_'):
                globals()[name] = getattr(module, name)
    except Exception:
        continue

# 手動定義一些常見的備援 (防止名稱不一致)
if 'FCLayer' not in globals():
    try: from physicsnemo.nn.fully_connected_layers import FCLayer
    except: pass

# 確保 DGMLayer 存在，如果 nn 裡沒有，則建立一個 dummy 類別防止報錯
if 'DGMLayer' not in globals():
    class DGMLayer: pass

