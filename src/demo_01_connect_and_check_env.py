# demo_01_connect_and_check_env.py
from pymilvus import MilvusClient, __version__ as pymilvus_version
import time

print(f"PyMilvus version: {pymilvus_version}")

# Milvus 服务连接信息
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

client = None
try:
    # 连接到 Milvus 服务
    # 使用 host 和 port 指定 Milvus 服务的地址
    print(f"\nAttempting to connect to Milvus service at {MILVUS_HOST}:{MILVUS_PORT}...")
    client = MilvusClient(host=MILVUS_HOST, port=MILVUS_PORT)

    # 可以通过 list_collections() 来确认连接是否成功
    # 如果服务刚刚启动可能需要一点时间才能响应
    time.sleep(2) # 等待 Milvus 服务完全就绪

    collections = client.list_collections()
    print(f"\nSuccessfully connected to Milvus service.")
    print(f"Existing collections: {collections}")

except Exception as e:
    print(f"\nFailed to connect to Milvus service: {e}")
    print("Please ensure your Milvus service is running and accessible at the specified host and port.")

finally:
    # 在 demo 结束时关闭客户端连接
    if client:
        client.close()
        print("Client connection closed.")

print("\nDemo 1 finished.")