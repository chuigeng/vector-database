# demo_08_drop_collection.py
from pymilvus import MilvusClient
import time

# Milvus 服务连接信息
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

# Collection 名称 (与 demo_02 保持一致)
COLLECTION_NAME = "document_embeddings_demo"

client = None
try:
    client = MilvusClient(host=MILVUS_HOST, port=MILVUS_PORT)
    print(f"Successfully connected to Milvus service at {MILVUS_HOST}:{MILVUS_PORT}")

    # 检查 Collection 是否存在
    if client.has_collection(collection_name=COLLECTION_NAME):
        print(f"\nCollection '{COLLECTION_NAME}' exists. Attempting to drop it...")

        # 在删除 Collection 之前，建议先释放它 (如果已加载到内存)
        # 检查 Collection 是否已加载 (不是必须的，但好习惯)
        try:
             # 尝试获取加载状态，如果 Collection 不存在或未加载会抛异常
             load_state = client.get_load_state(collection_name=COLLECTION_NAME)
             if load_state.get('state') in ['LoadState.Loading', 'LoadState.Loaded']:
                  print(f"Collection '{COLLECTION_NAME}' is loaded. Releasing it first.")
                  client.release_collection(collection_name=COLLECTION_NAME)
                  print(f"Collection '{COLLECTION_NAME}' released.")
                  # 等待释放完成 (可选)
                  time.sleep(1)
        except Exception as e:
             # 如果 Collection 不存在或未加载，忽略异常
             pass


        # 删除 Collection
        client.drop_collection(collection_name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' drop request sent for '{COLLECTION_NAME}'.")

        # 等待删除完成 (可选)
        print("Waiting for collection deletion...")
        while client.has_collection(collection_name=COLLECTION_NAME):
             time.sleep(1)
        print(f"Collection '{COLLECTION_NAME}' confirmed to be deleted.")

    else:
        print(f"\nCollection '{COLLECTION_NAME}' does not exist. Nothing to drop.")

except Exception as e:
    print(f"\nError dropping collection: {e}")

finally:
    if client:
        client.close()
        print("Client connection closed.")

print("\nDemo 8 finished.")