#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
人脸向量API模块
提供人脸向量数据的API接口，用于前端可视化展示
"""

import os
import numpy as np
import base64
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
from pymilvus import MilvusClient
import time

# 导入配置
from face_vectorization import (
    MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME,
    ID_FIELD_NAME, NAME_FIELD_NAME, PATH_FIELD_NAME, EMBEDDING_FIELD_NAME
)

# 创建FastAPI应用
app = FastAPI(title="人脸向量可视化API")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境应当限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 连接Milvus
try:
    client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")
    print("Milvus 连接成功!")
except Exception as e:
    print(f"Milvus 连接失败: {e}")
    raise

class FaceNode(BaseModel):
    """人脸节点模型"""
    id: str
    name: str
    image_path: str
    image_data: str  # Base64编码的图像数据
    vector: List[float]

class FaceEdge(BaseModel):
    """人脸边（相似度连接）模型"""
    source: str
    target: str
    similarity: float

class FaceGraph(BaseModel):
    """人脸图结构模型"""
    nodes: List[FaceNode]
    edges: List[FaceEdge]

def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """计算两个向量之间的余弦相似度"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return float(similarity)

def image_to_base64(image_path: str) -> str:
    """将图像转换为Base64编码"""
    try:
        # 如果图像路径不是绝对路径，则将其视为相对于当前文件的路径
        if not os.path.isabs(image_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            image_path = os.path.join(current_dir, image_path)
        
        print(f"尝试读取图像文件: {image_path}")
        
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            return base64.b64encode(img_data).decode('utf-8')
    except Exception as e:
        print(f"图像转换失败: {e}")
        # 返回空字符串或默认图像
        return ""

@app.get("/face-graph", response_model=FaceGraph)
async def get_face_graph(similarity_threshold: float = 0.7):
    """
    获取人脸向量图数据
    
    参数:
        similarity_threshold: 相似度阈值，只有超过此值的边才会被返回
        
    返回:
        FaceGraph: 人脸图数据，包含节点和边
    """
    try:
        # 查询所有人脸数据
        print(f"\nLoading collection '{COLLECTION_NAME}' into memory...")
        client.load_collection(collection_name=COLLECTION_NAME, repeatedly_load=False)
        print(f"Collection '{COLLECTION_NAME}' loaded or already loaded.")
        time.sleep(2)
        print("Collection loading complete.")

        results = client.query(
            collection_name=COLLECTION_NAME,
            filter="",
            limit=1000,
            output_fields=[ID_FIELD_NAME, NAME_FIELD_NAME, PATH_FIELD_NAME, EMBEDDING_FIELD_NAME]
        )
        
        if not results:
            raise HTTPException(status_code=404, detail="未找到人脸数据")
        
        # 转换为节点列表
        nodes = []
        for face in results:
            # 为图像路径添加完整路径
            image_path = face[PATH_FIELD_NAME]
            # 获取Base64图像数据
            print(f"image_path: {image_path}")
            image_data = image_to_base64(image_path)
            
            nodes.append(FaceNode(
                id=str(face[ID_FIELD_NAME]),
                name=face[NAME_FIELD_NAME],
                image_path=image_path,
                image_data=image_data,
                vector=face[EMBEDDING_FIELD_NAME]
            ))
        
        # 计算所有人脸之间的相似度
        edges = []
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i < j:  # 避免重复计算
                    similarity = (compute_cosine_similarity(node1.vector, node2.vector) - 0.8) * 5
                    # 只添加相似度超过阈值的边
                    if similarity >= similarity_threshold:
                        edges.append(FaceEdge(
                            source=node1.id,
                            target=node2.id,
                            similarity=similarity
                        ))
        return FaceGraph(nodes=nodes, edges=edges)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取人脸图数据失败: {str(e)}")

# 如果直接运行此文件
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100) 