import socket
import time
import struct
import math

# 配置 UDP
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"🚀 模擬感測器啟動，正將數據發往 {UDP_PORT}...")

t = 0
try:
    while True:
        # 模擬一個在 0.3 到 1.2 之間波動的數值 (類比你的 113.28W 熱模擬數據)
        simulated_data = 0.75 + 0.45 * math.sin(t)
        
        # 將浮點數打包成二進制發送
        payload = struct.pack('f', simulated_data)
        sock.sendto(payload, (UDP_IP, UDP_PORT))
        
        print(f"📡 發送即時數據: {simulated_data:.4f}")
        t += 0.1
        time.sleep(0.05) # 模擬 20Hz 的採樣率
except KeyboardInterrupt:
    print("🛑 停止模擬")