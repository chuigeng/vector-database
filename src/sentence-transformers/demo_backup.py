from pymilvus import connections, utility
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
from sentence_transformers import SentenceTransformer
import numpy as np
import time

# --- 0. 配置参数 ---
MILVUS_HOST = "localhost"
MILVUS_PORT = "19379"
COLLECTION_NAME = "sentence_transformer_demo_collection"
ID_FIELD_NAME = "id"
TEXT_FIELD_NAME = "original_text" # 存储原始文本，方便查看结果
EMBEDDING_FIELD_NAME = "embedding"

# Sentence Transformers 模型
MODEL_NAME = 'all-MiniLM-L6-v2' # 这是一个常用且效果不错的模型 (维度 384)
# MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2' # 多语言模型 (维度 384)

# --- 1. 连接 Milvus ---
try:
    print(f"正在连接 Milvus ({MILVUS_HOST}:{MILVUS_PORT})...")
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    print("Milvus 连接成功!")
except Exception as e:
    print(f"Milvus 连接失败: {e}")
    exit()

# --- 2. 加载 Sentence Transformers 模型并获取维度 ---
print(f"正在加载 Sentence Transformers 模型: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
EMBEDDING_DIM = model.get_sentence_embedding_dimension()
print(f"模型加载完毕。嵌入维度 (dim): {EMBEDDING_DIM}")

# --- 3. 定义 Schema ---
# 主键字段 (自动生成 ID)
field_id = FieldSchema(
    name=ID_FIELD_NAME,
    dtype=DataType.INT64,
    is_primary=True,
    auto_id=True, # 设置为 True，Milvus 会自动生成 ID
    description="Primary key with auto-incrementing IDs"
)
# 原始文本字段 (可选，但推荐用于调试和结果展示)
field_text = FieldSchema(
    name=TEXT_FIELD_NAME,
    dtype=DataType.VARCHAR,
    max_length=65535, # 根据你的文本最大长度设定
    description="Original text content"
)
# 向量字段
field_embedding = FieldSchema(
    name=EMBEDDING_FIELD_NAME,
    dtype=DataType.FLOAT_VECTOR,
    dim=EMBEDDING_DIM, # **dim 是必传的**
    description="Float vector embeddings from Sentence Transformers"
)

# 创建 Schema
schema = CollectionSchema(
    fields=[field_id, field_text, field_embedding],
    description="Collection for storing sentence embeddings",
    enable_dynamic_field=False # 通常对于结构化数据设为 False
)
print("Schema 定义完成。")

# --- 4. 创建集合 ---
# 先检查集合是否存在，如果存在则删除（方便重复运行脚本）
if utility.has_collection(COLLECTION_NAME):
    print(f"集合 '{COLLECTION_NAME}' 已存在，将删除后重建。")
    utility.drop_collection(COLLECTION_NAME)
    time.sleep(1) # 等待删除操作完成

try:
    collection = Collection(
        name=COLLECTION_NAME,
        schema=schema,
        using='default', # 使用默认的 alias
        consistency_level="Strong" # 或者 "Bounded", "Session", "Eventually"
    )
    print(f"集合 '{COLLECTION_NAME}' 创建成功!")
except Exception as e:
    print(f"集合创建失败: {e}")
    exit()

# --- 5. 准备并插入数据 ---
print("准备数据并生成嵌入...")
sentences_to_insert = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is rapidly changing the world.",
    "Milvus is a vector database for AI applications.",
    "Sentence Transformers provide easy-to-use text embeddings.",
    "Today is a sunny day, perfect for a walk in the park.",
    "What are the best practices for vector similarity search?",
    "Natural language processing enables computers to understand human language."
]

# 使用 Sentence Transformers 模型生成嵌入
embeddings = model.encode(sentences_to_insert, show_progress_bar=True)
# embeddings 是一个 numpy 数组，每一行是一个句子的嵌入

# 准备插入 Milvus 的数据
# PyMilvus 的 insert 方法接受一个 list of lists 或者 list of dicts
# 如果 schema 中 is_primary=True 且 auto_id=True，则不需要提供主键字段的值
# 数据顺序应与 schema 中定义的非 auto_id 字段顺序一致
data_to_insert_milvus = []
for i in range(len(sentences_to_insert)):
    # 顺序: [original_text, embedding]
    data_to_insert_milvus.append([sentences_to_insert[i], embeddings[i].tolist()])

print(f"生成了 {len(data_to_insert_milvus)} 条嵌入数据。")

# 插入数据
try:
    print("正在向 Milvus 插入数据...")
    insert_result = collection.insert(data_to_insert_milvus)
    print(f"数据插入成功! Inserted IDs: {insert_result.primary_keys}")
    print(f"成功插入 {insert_result.insert_count} 条实体。")

    # Flush 操作确保数据对后续操作可见
    collection.flush()
    print(f"数据已 Flush。当前集合实体数量: {collection.num_entities}")
except Exception as e:
    print(f"数据插入失败: {e}")
    exit()


# --- 6. 创建索引 ---
# 为向量字段创建索引以加速搜索
# 对于句子嵌入，COSINE (或 IP，如果向量已归一化) 通常是好的度量类型
# HNSW 是一个常用的高性能索引类型
INDEX_METRIC_TYPE = "COSINE" # 'L2' 或 'IP' (Inner Product) 或 'COSINE'
                               # Sentence Transformers (like all-MiniLM-L6-v2) often produce normalized embeddings,
                               # making IP equivalent to COSINE. COSINE is generally recommended for semantic similarity.
index_params = {
    "metric_type": INDEX_METRIC_TYPE,
    "index_type": "HNSW", # 常用的ANN索引类型
    "params": {
        "M": 16,           # HNSW 的图连接度 (典型值 4-64)
        "efConstruction": 200  # HNSW 构建时的搜索范围 (典型值 M 的数倍)
    }
}

try:
    print(f"正在为字段 '{EMBEDDING_FIELD_NAME}' 创建索引...")
    collection.create_index(
        field_name=EMBEDDING_FIELD_NAME,
        index_params=index_params
    )
    print("索引创建请求已发送。等待索引构建完成...")
    utility.wait_for_index_building_complete(COLLECTION_NAME)
    print("索引构建完成!")
    # 打印索引信息
    print("\n当前集合的索引信息:")
    for index in collection.indexes:
        print(f"  字段: {index.field_name}, 索引名称: {index.index_name}, 参数: {index.params}")

except Exception as e:
    print(f"索引创建失败: {e}")
    exit()

# --- 7. 加载集合到内存 ---
# 在搜索之前，需要将集合加载到内存中
try:
    print("正在加载集合到内存...")
    collection.load()
    utility.wait_for_loading_complete(COLLECTION_NAME)
    print("集合加载完成!")
except Exception as e:
    print(f"集合加载失败: {e}")
    exit()


# --- 8. 执行向量相似度搜索 ---
query_sentence = "What is a vector DB?"
print(f"\n准备查询: \"{query_sentence}\"")

# 1. 将查询语句转换为向量
query_embedding = model.encode([query_sentence])[0].tolist() # model.encode 返回一个列表的列表或numpy数组

# 2. 定义搜索参数
TOP_K = 3 # 返回最相似的 top_k 个结果
SEARCH_PARAMS_HNSW = {
    "metric_type": INDEX_METRIC_TYPE, # 必须与索引创建时一致或兼容
    "params": {
        "ef": 128  # HNSW 搜索时的探索范围，ef >= top_k，通常比 top_k 大很多
    }
}

print(f"正在执行搜索 (Top K={TOP_K})...")
try:
    search_results = collection.search(
        data=[query_embedding],             # 查询向量，可以批量查询，所以是列表
        anns_field=EMBEDDING_FIELD_NAME,    # 要搜索的向量字段名
        param=SEARCH_PARAMS_HNSW,           # 搜索参数，对应索引类型
        limit=TOP_K,                        # 返回结果数量
        expr=None,                          # 可选的标量字段过滤表达式
        output_fields=[TEXT_FIELD_NAME, ID_FIELD_NAME], # 希望返回的字段，除了距离和主键ID
        consistency_level="Strong"          # 搜索时的一致性级别
    )

    # 3. 处理并打印搜索结果
    print(f"\n--- 搜索结果 (与 \"{query_sentence}\" 最相似的 {TOP_K} 条): ---")
    for i, hits in enumerate(search_results): # search_results 是一个列表，对应每个查询向量的结果
        print(f"  查询 {i+1} 的结果:")
        if not hits:
            print("    未找到结果。")
            continue
        for hit in hits:
            # hit 对象包含: id, distance, entity (如果指定了 output_fields)
            # hit.entity 是一个字典，键是 output_fields 中指定的字段名
            original_text = hit.entity.get(TEXT_FIELD_NAME, "N/A") if hit.entity else "N/A"
            print(f"    - ID: {hit.id}, 距离(Distance/Score): {hit.distance:.4f}, 文本: \"{original_text}\"")

except Exception as e:
    print(f"搜索失败: {e}")


# --- 9. 清理 (可选) ---
# 如果想在脚本结束时删除集合
# print(f"\n正在删除集合 '{COLLECTION_NAME}'...")
# utility.drop_collection(COLLECTION_NAME)
# print("集合已删除。")

print("\nDemo 执行完毕。")