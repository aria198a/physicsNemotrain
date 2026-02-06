import torch

def signed_distance_field(invar, geometry, params=None, compute_sdf_derivatives=False, **kwargs):
    # 獲取座標 (處理 Dict 或 Tensor)
    if isinstance(invar, dict):
        x, y, z = invar.get('x'), invar.get('y'), invar.get('z')
    else:
        x, y, z = invar[:, 0:1], invar[:, 1:2], invar[:, 2:3]

    # 【驗證邏輯】：判斷點是否在一個 0.2m 的立方體內 (對應你縮放後的 STL)
    # 如果比例大於 0，代表採樣點有打到這個範圍
    mask = (x.abs() < 0.11) & (y.abs() < 0.11) & (z.abs() < 0.05)
    sdf_values = torch.where(mask, torch.ones_like(x), -torch.ones_like(x))
    
    sdf_derivative = torch.zeros((x.shape[0], 1, 3), device=x.device)
    return sdf_values, sdf_derivative
