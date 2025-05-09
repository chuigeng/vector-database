# test_milvus.py
from pymilvus import connections, utility

try:
    print("准备连接到 Milvus Lite...")
    connections.connect(alias="default", host="localhost", port="19530")
    print("成功连接到 Milvus Lite!")
    print("服务器版本:", utility.get_server_version())
    connections.disconnect(alias="default")
except Exception as e:
    print(f"连接失败: {e}")