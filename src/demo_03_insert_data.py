# demo_03_insert_data.py
import random
import time
from pymilvus import MilvusClient

# Milvus 服务连接信息
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

# Collection 名称 (与 demo_02 保持一致)
COLLECTION_NAME = "document_embeddings_demo"
# 向量维度 (与 demo_02 保持一致)
VECTOR_DIM = 8

client = None
try:
    client = MilvusClient(host=MILVUS_HOST, port=MILVUS_PORT)
    print(f"Successfully connected to Milvus service at {MILVUS_HOST}:{MILVUS_PORT}")

    # 检查 Collection 是否存在
    if not client.has_collection(collection_name=COLLECTION_NAME):
        print(f"Error: Collection '{COLLECTION_NAME}' not found. Run demo_02 first.")
        exit()

    # 准备要插入的数据 (Entities)
    # 注意：doc_id 是自动生成的，所以这里不需要提供
    entities = [
        {"category": "Technology", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Science", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Arts", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Sports", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Business", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Health", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Entertainment", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Politics", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Education", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Travel", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Technology", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Science", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Arts", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Sports", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Business", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Health", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Entertainment", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Politics", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Education", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Travel", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Technology", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Science", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Arts", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Sports", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Business", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Health", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Entertainment", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Politics", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Education", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Travel", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Technology", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Science", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Arts", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Sports", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Business", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Health", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Entertainment", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Politics", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Education", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
        {"category": "Travel", "embedding": [random.random() * 20 for _ in range(VECTOR_DIM)]},
    ]

    print(f"\nAttempting to insert {len(entities)} entities into '{COLLECTION_NAME}'...")

    # 插入数据
    insert_result = client.insert(collection_name=COLLECTION_NAME, data=entities)

    print("Insert result:", insert_result)
    # 插入结果包含自动生成的 ID

    # 刷新数据，使其可被搜索和查询
    # 在 Milvus 中，插入的数据首先进入内存缓冲区，flush 操作将其持久化并使其可查询
    print("\nFlushing data...")
    client.flush(collection_name=COLLECTION_NAME)
    print("Flush complete.")

    # 等待数据实际写入
    time.sleep(2) # 根据服务负载可能需要调整等待时间

    # 获取 Collection 状态，查看 Entity 数量
    stats = client.get_collection_stats(collection_name=COLLECTION_NAME)
    print(f"\nCollection '{COLLECTION_NAME}' stats after insert and flush: {stats}")


except Exception as e:
    print(f"\nError inserting data: {e}")

finally:
    if client:
        client.close()
        print("Client connection closed.")

print("\nDemo 3 finished.")