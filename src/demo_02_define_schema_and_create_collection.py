# demo_02_define_schema_and_create_collection.py
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType

# Milvus 服务连接信息
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

# Collection 名称
COLLECTION_NAME = "document_embeddings_demo"
# 向量维度 (与 demo_03, demo_05, demo_06 保持一致)
VECTOR_DIM = 8

client = None
try:
    client = MilvusClient(host=MILVUS_HOST, port=MILVUS_PORT)
    print(f"Successfully connected to Milvus service at {MILVUS_HOST}:{MILVUS_PORT}")

    # 如果 Collection 已存在，先删除以便重新创建 (用于演示目的)
    if client.has_collection(collection_name=COLLECTION_NAME):
        print(f"\nCollection '{COLLECTION_NAME}' already exists. Dropping it.")
        client.drop_collection(collection_name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' dropped.")

    # 1. 定义 Fields (字段)
    fields = [
        # 主键字段：doc_id，整型，自动生成 ID
        FieldSchema(name="doc_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        # 标量字段：category，字符串类型，用于过滤
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=256),
        # 向量字段：embedding，浮点向量，指定维度
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    ]

    # 2. 定义 Collection 的 Schema
    schema = CollectionSchema(fields=fields, description="Demo collection for document embeddings")

    # 3. 创建 Collection
    print(f"\nAttempting to create collection '{COLLECTION_NAME}'...")
    client.create_collection(collection_name=COLLECTION_NAME, schema=schema)

    print(f"Collection '{COLLECTION_NAME}' created successfully with schema:")
    print(schema)

except Exception as e:
    print(f"\nError creating collection: {e}")

finally:
    if client:
        client.close()
        print("Client connection closed.")

print("\nDemo 2 finished.")