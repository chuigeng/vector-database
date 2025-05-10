from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
from pymilvus import model
import time

# --- 0. 配置参数 ---
MILVUS_HOST = "localhost"
MILVUS_PORT = "19379"
COLLECTION_NAME = "sentence_transformer_demo_collection"
ID_FIELD_NAME = "id"
TEXT_FIELD_NAME = "original_text" # 存储原始文本，方便查看结果
EMBEDDING_FIELD_NAME = "embedding"

# Sentence Transformers 模型
# MODEL_NAME = 'all-MiniLM-L6-v2' # 这是一个常用且效果不错的模型 (维度 384)
# MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2' # 多语言模型 (维度 384)
MODEL_NAME = 'BAAI/bge-large-zh-v1.5' # 中文特训模型
DEVICE = 'cpu' # Specify the device to use, e.g., 'cpu' or 'cuda:0'

# --- 1. 连接 Milvus ---
client = None
try:
    print(f"正在连接 Milvus ({MILVUS_HOST}:{MILVUS_PORT})...")
    client = MilvusClient(host=MILVUS_HOST, port=MILVUS_PORT)
    print("Milvus 连接成功!")
except Exception as e:
    print(f"Milvus 连接失败: {e}")
    exit()

# --- 2. 加载 Sentence Transformers 模型并获取维度 ---
print(f"正在加载 Sentence Transformers 模型: {MODEL_NAME}...")
model = model.dense.SentenceTransformerEmbeddingFunction(
    model_name=MODEL_NAME, # Specify the model name
    device=DEVICE
)
EMBEDDING_DIM = model.dim
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
if client.has_collection(collection_name=COLLECTION_NAME):
    print(f"集合 '{COLLECTION_NAME}' 已存在，将删除后重建。")
    client.drop_collection(collection_name=COLLECTION_NAME)
    time.sleep(1) # 等待删除操作完成

try:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema
    )
    print(f"集合 '{COLLECTION_NAME}' 创建成功!")
except Exception as e:
    print(f"集合创建失败: {e}")
    exit()

# --- 5. 准备并插入数据 ---
print("准备数据并生成嵌入...")
sentences_to_insert = [
    "敏捷的棕色狐狸跳过了懒惰的狗。",
    "人工智能正在迅速改变世界。",
    "Milvus 是一个用于 AI 应用的向量数据库。",
    "Sentence Transformers 提供了易于使用的文本嵌入。",
    "今天是个阳光明媚的日子，非常适合去公园散步。",
    "向量相似度搜索的最佳实践是什么？",
    "自然语言处理使计算机能够理解人类语言。"
]

# 使用 Sentence Transformers 模型生成嵌入
embeddings = model.encode_documents(sentences_to_insert)
# embeddings 是一个 numpy 数组，每一行是一个句子的嵌入

# 准备插入 Milvus 的数据
# MilvusClient 的插入方法需要字典格式的数据
data_to_insert_milvus = []
for i in range(len(sentences_to_insert)):
    data_to_insert_milvus.append({
        TEXT_FIELD_NAME: sentences_to_insert[i], 
        EMBEDDING_FIELD_NAME: embeddings[i].tolist()
    })

print(f"生成了 {len(data_to_insert_milvus)} 条嵌入数据。")

# 插入数据
try:
    print("正在向 Milvus 插入数据...")
    insert_result = client.insert(
        collection_name=COLLECTION_NAME,
        data=data_to_insert_milvus
    )
    print(f"数据插入成功!")
    print(f"成功插入 {len(data_to_insert_milvus)} 条实体。")

    # 刷新数据，使其可被搜索和查询
    # 在 Milvus 中，插入的数据首先进入内存缓冲区，flush 操作将其持久化并使其可查询
    print("\nFlushing data...")
    client.flush(collection_name=COLLECTION_NAME)
    print("Flush complete.")
    
    # 获取集合实体数量
    stats = client.get_collection_stats(collection_name=COLLECTION_NAME)
    print(f"当前集合实体数量: {stats['row_count']}")
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
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=EMBEDDING_FIELD_NAME,
        index_type="HNSW",
        metric_type=INDEX_METRIC_TYPE,
        params={"M": 16, "efConstruction": 200},
        index_name="sentence_transformer_demo_index"
    )
    client.create_index(
        collection_name=COLLECTION_NAME,
        index_params=index_params
    )
    print("索引创建完成!")
    # 打印索引信息
    print("\n当前集合的索引信息:")
    indexes = client.describe_index(collection_name=COLLECTION_NAME, index_name="sentence_transformer_demo_index")
    for index in indexes:
        print(f"  字段: {index}: {indexes.get(index)}")

except Exception as e:
    print(f"索引创建失败: {e}")
    exit()

# --- 7. 加载集合到内存 ---
# 在搜索之前，需要将集合加载到内存中
try:
    print("正在加载集合到内存...")
    client.load_collection(collection_name=COLLECTION_NAME)
    time.sleep(2)
    print("集合加载完成!")
except Exception as e:
    print(f"集合加载失败: {e}")
    exit()


# --- 8. 执行向量相似度搜索 ---
query_sentence = "什么是向量数据库?"
print(f"\n准备查询: \"{query_sentence}\"")

# 1. 将查询语句转换为向量
query_embedding = model.encode_queries([query_sentence])[0].tolist() # model.encode 返回一个列表的列表或numpy数组

# 2. 定义搜索参数
TOP_K = 3 # 返回最相似的 top_k 个结果

print(f"正在执行搜索 (Top K={TOP_K})...")
try:
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_embedding],             # 查询向量，可以批量查询，所以是列表
        anns_field=EMBEDDING_FIELD_NAME,    # 要搜索的向量字段名
        params={"ef": 128},                  # HNSW 搜索时的探索范围，ef >= top_k，通常比 top_k 大很多           
        limit=TOP_K,                        # 返回结果数量
        output_fields=[TEXT_FIELD_NAME]     # 希望返回的字段，除了距离和主键ID
    )

    # 3. 处理并打印搜索结果
    print(f"\n--- 搜索结果 (与 \"{query_sentence}\" 最相似的 {TOP_K} 条): ---")
    if not search_results:
        print("    未找到结果。")
    else:
        for i, hits in enumerate(search_results):
            print(f"  查询 {i+1} 的结果:")
            for hit in hits:
                print(f"    - hit: {hit}")

except Exception as e:
    print(f"搜索失败: {e}")


# --- 9. 清理 (可选) ---
# 如果想在脚本结束时删除集合
# print(f"\n正在删除集合 '{COLLECTION_NAME}'...")
# client.drop_collection(collection_name=COLLECTION_NAME)
# print("集合已删除。")

# 关闭客户端连接
if client:
    client.close()
    print("客户端连接已关闭。")

print("\nDemo 执行完毕。") 