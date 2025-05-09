# demo_07_get_data_by_ids.py
import time
from pymilvus import MilvusClient

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
    if not client.has_collection(collection_name=COLLECTION_NAME):
        print(f"Error: Collection '{COLLECTION_NAME}' not found. Run demo_02 and demo_03 first.")
        exit()

     # 确保 Collection 已加载 (Get 操作通常也需要加载)
    print(f"\nLoading collection '{COLLECTION_NAME}' into memory...")
    client.load_collection(collection_name=COLLECTION_NAME, repeatedly_load=False)
    print(f"Collection '{COLLECTION_NAME}' loaded or already loaded.")
    time.sleep(2)
    print("Collection loading complete.")

    # 1. 获取 Collection 中存在的 ID (为了演示，这里先执行一个简单的 Query 来获取一些 ID)
    # 实际应用中，你可能已经知道要获取的 ID 列表
    print("\nGetting some IDs by querying for 'Technology'...")
    query_results = client.query(
        collection_name=COLLECTION_NAME,
        filter='category == "Technology"',
        output_fields=["doc_id"]
    )

    ids_to_get = [entity['doc_id'] for entity in query_results]
    print(f"Found IDs to get: {ids_to_get}")

    if not ids_to_get:
        print("No IDs found to retrieve. Ensure demo_03 inserted data with category 'Technology'.")
        exit()

    # 2. 根据 ID 获取 Entity 数据
    print(f"\nAttempting to get entities by IDs: {ids_to_get}")
    get_results = client.get(
        collection_name=COLLECTION_NAME,
        ids=ids_to_get,
        output_fields=["doc_id", "category", "embedding"] # 可以指定要返回的字段，包括向量字段
    )

    print("\nGet results:")
    if get_results:
        for entity in get_results:
             # 注意：embedding 字段可能很长，这里简化打印
            print(f"  ID: {entity['doc_id']}, Category: {entity['category']}, Embedding (partial): {entity['embedding'][:5]}...")
    else:
        print("  No entities retrieved for the given IDs.")

except Exception as e:
    print(f"\nError getting data by IDs: {e}")

finally:
     # 完成操作后释放 Collection (可选)
    # print(f"\nReleasing collection '{COLLECTION_NAME}' from memory...")
    # client.release_collection(collection_name=COLLECTION_NAME)
    # print(f"Collection '{COLLECTION_NAME}' released.")

    if client:
        client.close()
        print("Client connection closed.")

print("\nDemo 7 finished.")