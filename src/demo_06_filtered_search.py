# demo_06_filtered_search.py
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

    # 确保 Collection 已加载
    print(f"\nLoading collection '{COLLECTION_NAME}' into memory...")
    client.load_collection(collection_name=COLLECTION_NAME, repeatedly_load=False)
    print(f"Collection '{COLLECTION_NAME}' loaded or already loaded.")
    time.sleep(2)
    print("Collection loading complete.")


    # 1. 准备查询向量
    query_vector = [random.random() for _ in range(VECTOR_DIM)]
    print(f"\nQuery vector: {query_vector}")

    # 2. 执行带过滤条件的向量搜索 (Filtered Search)
    # filter 参数是一个布尔表达式，使用标量字段进行过滤
    print("\nPerforming filtered vector search (category == 'Technology')...")

    search_results_filtered = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],           # 查询向量列表
        filter='category == "Technology"', # 只搜索 category 为 "Technology" 的实体
        limit=3,                       # 返回 Top 3 结果
        output_fields=["category"],    # 返回 category 字段
        anns_field=VECTOR_FIELD_NAME,  # 指定在哪个向量字段上搜索
        metric_type="L2",             # 距离计算方式
        params={"nprobe": 10}          # 搜索参数
    )

    print("\nFiltered search results:")
    if search_results_filtered and search_results_filtered[0]:
        for hit in search_results_filtered[0]:
            print(f"  ID: {hit}")
    else:
        print("  No results found matching the filter.")

    # 3. 也可以执行只基于过滤的查询 (Query 操作)
    # Query 不需要向量，只根据标量过滤条件查找数据
    print("\nPerforming scalar query (category == 'Science')...")
    query_results = client.query(
        collection_name=COLLECTION_NAME,
        filter='category == "Science"',
        output_fields=["doc_id", "category"] # 指定返回的字段
    )

    print("\nQuery results:")
    if query_results:
        for entity in query_results:
            print(f"  ID: {hit}")
    else:
         print("  No entities found matching the query filter.")


except Exception as e:
    print(f"\nError during filtered search or query: {e}")

finally:
     # 完成操作后释放 Collection (可选)
    # print(f"\nReleasing collection '{COLLECTION_NAME}' from memory...")
    # client.release_collection(collection_name=COLLECTION_NAME)
    # print(f"Collection '{COLLECTION_NAME}' released.")

    if client:
        client.close()
        print("Client connection closed.")

print("\nDemo 6 finished.")