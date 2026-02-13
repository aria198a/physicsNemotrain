from physicsnemo.nn import FCLayer, get_activation
# 如果它還缺 Conv1dFCLayer，我們先用 FCLayer 頂替或從 nn 尋找
try:
    from physicsnemo.nn import Conv1dFCLayer
except ImportError:
    Conv1dFCLayer = FCLayer 
