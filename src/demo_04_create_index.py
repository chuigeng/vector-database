# demo_04_create_index.py
from pymilvus import MilvusClient

# Milvus 服务连接信息
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

# Collection 名称 (与 demo_02 保持一致)
COLLECTION_NAME = "document_embeddings_demo"
# 向量字段名称 (与 demo_02 保持一致)
VECTOR_FIELD_NAME = "embedding"
# 索引名称 (可以自定义)
INDEX_NAME = "my_vector_index"

client = None
try:
    client = MilvusClient(host=MILVUS_HOST, port=MILVUS_PORT)
    print(f"Successfully connected to Milvus service at {MILVUS_HOST}:{MILVUS_PORT}")

    # 检查 Collection 是否存在
    if not client.has_collection(collection_name=COLLECTION_NAME):
        print(f"Error: Collection '{COLLECTION_NAME}' not found. Run demo_02 first.")
        exit()

    # 检查是否已存在同名索引，如果存在则删除 (用于演示目的)
    try:
        index_info = client.describe_index(collection_name=COLLECTION_NAME, index_name=INDEX_NAME)
        if index_info:
             print(f"\nIndex '{INDEX_NAME}' already exists. Dropping it.")
             client.drop_index(collection_name=COLLECTION_NAME, index_name=INDEX_NAME)
             print(f"Index '{INDEX_NAME}' dropped.")
    except Exception as e:
        # describe_index 在索引不存在时会抛异常，这里忽略
        pass


    # 1. 定义索引参数
    # 这里使用 IVF_FLAT 索引类型作为示例，它是一种常见的聚类索引
    # metric_type 指定相似度度量 (L2, IP, COSINE)
    # params 是索引类型的特定参数，nlist 是 IVF 索引的聚类数量
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=VECTOR_FIELD_NAME,
        index_type="IVF_FLAT", # 簇，搜索时只在查询向量最近的几个簇中进行
        # index_type="HNSW", # 基于图的索引，搜索时在所有向量中进行
        metric_type="COSINE",       # 相似度度量
        index_name=INDEX_NAME,  # 索引名称
        # 索引参数，指定要使用 k-means 算法创建的分区数，和 nprobe 配合使用，指定在搜索候选对象期间要考虑的分区数
        # https://milvus.io/docs/ivf-flat.md#Overview
        # https://milvus.io/docs/performance_faq.md#How-to-set-nlist-and-nprobe-for-IVF-indexes
        params={"nlist": 128}

        # 使用 HNSW 索引
        # M 定义了在 HNSW 图的构建过程中，每个节点（除了第0层，通常是 2*M）在每一层上允许拥有的最大出度（outgoing connections / neighbors）。简单来说，它控制了图中每个节点的“邻居”数量的上限。
        # M 的选择是一个权衡。较低的 M 会导致更稀疏的图，构建更快，内存更少，但可能需要更高的 ef (搜索时参数) 才能达到好的召回率。较高的 M 会构建更稠密的图，构建更慢，内存更多，但通常图的质量更好，可能在搜索时用较低的 ef 就能达到高召回
        # efConstruction 控制了在图构建过程中，为新插入的节点寻找其 M 个最近邻居时，搜索的“深度”或“广度”。具体来说，它是在构建索引、插入新节点时，动态维护的候选邻居列表的大小。算法会从这个候选列表中选择最好的 M 个邻居进行连接。
        # efConstruction 值越大，在插入新节点时，算法会探索更广泛的邻居候选，从而更有可能找到真正好的连接，避免陷入局部最优，因此构建出的图的质量通常更高。高质量的图对于后续的搜索性能至关重要。
        # 搭配查询参数 ef 使用
        # HNSW 的搜索机制回顾：
        #     HNSW 索引是一个多层图结构。搜索从顶层的一个（或多个）入口点开始。
        #     在每一层，算法会贪婪地走向查询向量的最近邻居。
        #     为了找到这些邻居，算法会维护一个动态的、按与查询向量距离排序的候选列表（通常是优先队列）。
        #     算法会从这个候选列表中取出最近的未访问节点进行探索，并将其邻居加入候选列表。
        #     这个过程会持续进行，直到找不到比当前已找到的 top_k 个结果更好的候选者，或者满足其他停止条件。
        # ef 的角色：
        #     ef 参数直接决定了这个动态候选列表允许的最大容量。
        #     当算法在图中导航时，它会评估遇到的节点，并将它们（如果足够好）放入这个候选列表中。
        #     如果候选列表已满（达到 ef 的大小）并且来了一个新的、比列表中最差的候选者更好的节点，那么最差的候选者就会被移除，新的节点被加入。
        # ef 的影响：
        #     召回率 (Recall)：
        #     较大的 ef：意味着搜索算法会考虑更多的潜在路径和候选节点。这增加了找到真正最近邻居的可能性，从而提高召回率。搜索会更“彻底”。
        #     较小的 ef：搜索范围更窄，可能会过早地停止探索，从而可能错过一些真正的近邻，导致召回率降低。
        # params={    
        #     "M": 64,
        #     "efConstruction": 100
        # }

    )

    # 2. 创建索引
    print(f"\nAttempting to create index '{INDEX_NAME}' on field '{VECTOR_FIELD_NAME}'...")
    # index_name 参数在 create_index 中是可选的，如果 index_params 中已指定则会使用它
    client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)

    print(f"Index creation request sent for '{INDEX_NAME}'.")

    # 可以描述索引信息来确认
    index_desc = client.describe_index(collection_name=COLLECTION_NAME, index_name=INDEX_NAME)
    print("\nIndex description:")
    print(index_desc)


except Exception as e:
    print(f"\nError creating index: {e}")

finally:
    if client:
        client.close()
        print("Client connection closed.")

print("\nDemo 4 finished.")