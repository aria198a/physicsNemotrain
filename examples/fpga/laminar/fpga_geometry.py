from sympy import Symbol
from physicsnemo.sym.geometry.primitives_3d import Channel, Plane, Box
from physicsnemo.sym.geometry import Parameterization, Parameter
from physicsnemo.sym.geometry.tessellation import Tessellation

# =========================================================
# ⚙️ 模式切換：'STL' (原始 40KW) 或 'ANALYTICAL' (簡化方塊)
# =========================================================
MODE = 'STL' 

# =========================================================
# 🔧 Domain 尺度設定 (公尺 m)
# =========================================================
# 風道區域 (Channel)
channel_origin = (-2.5, -0.5, -0.6)
channel_dim = (5.0, 1.0, 1.2)

# =========================================================
# 🔣 SymPy symbols
# =========================================================
x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

# =========================================================
# 🧊 Channel (流體母域)
# =========================================================
channel = Channel(
    channel_origin,
    (
        channel_origin[0] + channel_dim[0],
        channel_origin[1] + channel_dim[1],
        channel_origin[2] + channel_dim[2],
    ),
)

# =========================================================
# 🔩 Solid Geometry 邏輯
# =========================================================
if MODE == 'STL':
    print("[INFO] Loading Original 40KW STL Geometry...")
    # 載入 89年次屬龍的你所使用的原始設計
    solid = Tessellation.from_stl(
        "/home/os-i-jingtai.chang/PhysicsNemo/physicsnemo-sym/examples/fpga/laminar/solid_40KW.stl",
        airtight=True
    )
    # 關鍵對齊：從 mm 縮放至 m
    solid = solid.scale(0.001)
    # 平移對齊：確保散熱片中心位於風道原點
    solid = solid.translate((0.0, 0.0, 0.0))

else:
    print("[INFO] Building Analytical Simplified Geometry...")
    # 建立簡化版散熱體 (用於對比訓練差異)
    base = Box(point_1=(-0.1, -0.1, -0.01), point_2=(0.1, 0.1, 0.01))
    fins = []
    for i in range(10):
        fin = Box(
            point_1=(-0.09 + i*0.02, -0.1, 0.01), 
            point_2=(-0.085 + i*0.02, 0.1, 0.06)
        )
        fins.append(fin)
    solid = base
    for f in fins: solid = solid + f

# =========================================================
# 🌊 最終流體域：geo = channel - solid
# =========================================================
# 此 geo 將用於 fpga_flow.py 與 fpga_heat.py 的 PDE 損失計算
geo = channel - solid

# =========================================================
# 🚪 邊界定義 (Inlet / Outlet)
# =========================================================
inlet = Plane(
    channel_origin,
    (channel_origin[0], channel_origin[1] + channel_dim[1], channel_origin[2] + channel_dim[2]),
    normal=-1,
)

outlet = Plane(
    (channel_origin[0] + channel_dim[0], channel_origin[1], channel_origin[2]),
    (channel_origin[0] + channel_dim[0], channel_origin[1] + channel_dim[1], channel_origin[2] + channel_dim[2]),
    normal=1,
)

print(f"✅ fpga_geometry.py loaded in {MODE} mode.")