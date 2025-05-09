# demo_05_load_collection_and_vector_search.py
import random
import time
from pymilvus import MilvusClient

# Milvus 服务连接信息
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

# Collection 名称 (与 demo_02 保持一致)
COLLECTION_NAME = "document_embeddings_demo"
# 向量字段名称 (与 demo_02 保持一致)
VECTOR_FIELD_NAME = "embedding"
# 向量维度 (与 demo_02 保持一致)
VECTOR_DIM = 8

client = None
try:
    client = MilvusClient(host=MILVUS_HOST, port=MILVUS_PORT)
    print(f"Successfully connected to Milvus service at {MILVUS_HOST}:{MILVUS_PORT}")

    # 检查 Collection 是否存在
    if not client.has_collection(collection_name=COLLECTION_NAME):
        print(f"Error: Collection '{COLLECTION_NAME}' not found. Run demo_02, demo_03, demo_04 first.")
        exit()

    # 1. 加载 Collection 到内存
    # 在执行搜索或查询操作之前，必须先加载 Collection
    print(f"\nLoading collection '{COLLECTION_NAME}' into memory...")
    # 如果 Collection 已经在内存中，repeatedly_load=False 可以避免重复加载
    client.load_collection(collection_name=COLLECTION_NAME, repeatedly_load=False)
    print(f"Collection '{COLLECTION_NAME}' loaded or already loaded.")

    # 等待短暂时间，确保集合加载完成
    print("Waiting for collection to load...")
    time.sleep(2)
    print("Collection loading complete.")


    # 2. 准备查询向量
    # 生成一个随机向量作为查询示例
    query_vector = [0.8254770928616384, 0.5083149379078975, 0.24583491203566776, 0.4446965696067242, 0.2561693833678802, 0.8406928582774604, 0.11623295292754388, 0.5535467904127089]
    print(f"\nQuery vector: {query_vector}")

    # 3. 执行向量相似度搜索
    print("\nPerforming vector search...")
    
    # 在新版 pymilvus 中正确的调用方式
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],           # 查询向量列表
        limit=3,                       # 返回 Top 3 结果
        output_fields=["category"],    # 返回 category 字段
        anns_field=VECTOR_FIELD_NAME,  # 指定在哪个向量字段上搜索
        params={"nprobe": 10}          # 搜索参数
    )

    print("\nSearch results:")
    # search_results 是一个列表，每个元素对应一个查询向量 (这里只有一个)
    if search_results and search_results[0]:
        for hit in search_results[0]:
            print(f"  ID: {hit}")
    else:
        print("  No search results found.")


except Exception as e:
    print(f"\nError during load or search: {e}")

finally:
    # 搜索完成后，可以释放 Collection 以节省内存 (可选，特别是在资源有限的环境中)
    # print(f"\nReleasing collection '{COLLECTION_NAME}' from memory...")
    # client.release_collection(collection_name=COLLECTION_NAME)
    # print(f"Collection '{COLLECTION_NAME}' released.")

    if client:
        client.close()
        print("Client connection closed.")

print("\nDemo 5 finished.")