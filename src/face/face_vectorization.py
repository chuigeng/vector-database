#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
人脸向量化处理模块
此脚本实现了提取人脸图像的特征向量并存储到Milvus向量数据库的功能
"""

import os
import numpy as np
import cv2
from tqdm import tqdm
import glob
import face_recognition
from pymilvus import (
    MilvusClient,
    FieldSchema, CollectionSchema, DataType,
    connections, Collection, utility
)
import time

# 配置参数
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "face_embeddings_collection"

# 字段名配置
ID_FIELD_NAME = "id"
NAME_FIELD_NAME = "name"
PATH_FIELD_NAME = "image_path"
EMBEDDING_FIELD_NAME = "embedding"
IMAGE_FIELD_NAME = "image_data"

# 向量维度 (face_recognition生成的面部特征向量是128维)
EMBEDDING_DIM = 128

class FaceVectorizer:
    def __init__(self, image_dir="src/face/image"):
        """
        初始化人脸向量化器
        
        参数:
            image_dir: 包含人脸图像的目录
        """
        self.image_dir = image_dir
        self.client = None
        self.connect_milvus()
    
    def connect_milvus(self):
        """连接到Milvus服务器"""
        try:
            print(f"正在连接 Milvus ({MILVUS_HOST}:{MILVUS_PORT})...")
            self.client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")
            print("Milvus 连接成功!")
        except Exception as e:
            print(f"Milvus 连接失败: {e}")
            raise
    
    def create_collection(self):
        """创建Milvus集合，如果不存在的话"""
        # 检查集合是否存在
        if self.client.has_collection(COLLECTION_NAME):
            print(f"集合 '{COLLECTION_NAME}' 已存在，正在删除...")
            self.client.drop_collection(COLLECTION_NAME)
            print(f"集合 '{COLLECTION_NAME}' 已删除。")

        # 1. 定义 Fields (字段)
        fields = [
            # 主键字段：doc_id，整型，自动生成 ID
            FieldSchema(name=ID_FIELD_NAME, dtype=DataType.INT64, is_primary=True, auto_id=True),
            # 标量字段：category，字符串类型，用于过滤
            FieldSchema(name=NAME_FIELD_NAME, dtype=DataType.VARCHAR, max_length=256),
            # 标量字段：category，字符串类型，用于过滤
            FieldSchema(name=PATH_FIELD_NAME, dtype=DataType.VARCHAR, max_length=256),
            # 向量字段：embedding，浮点向量，指定维度
            FieldSchema(name=EMBEDDING_FIELD_NAME, dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        ]

        # 2. 定义 Collection 的 Schema
        schema = CollectionSchema(fields=fields, description="人脸特征向量集合")
        
        try:
            self.client.create_collection(collection_name=COLLECTION_NAME, schema=schema);
            print(f"集合 '{COLLECTION_NAME}' 创建成功!")
            
            # 为向量字段创建索引
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name=EMBEDDING_FIELD_NAME,
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 200},
                index_name="face_embeddings_index"
            )
            
            self.client.create_index(
                collection_name=COLLECTION_NAME,
                index_params=index_params
            )
            print(f"向量索引创建完成!")
            
        except Exception as e:
            print(f"创建集合失败: {e}")
            raise
    
    def extract_face_encoding(self, image_path):
        """
        从图像中提取人脸编码
        
        参数:
            image_path: 图像文件路径
            
        返回:
            face_encoding: 人脸特征向量 (如果没有检测到人脸则返回None)
            face_location: 人脸位置坐标
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return None, None
        
        # 转换为RGB (face_recognition需要RGB格式)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 检测人脸位置
        face_locations = face_recognition.face_locations(rgb_image)
        
        if not face_locations:
            print(f"未在图像中检测到人脸: {image_path}")
            return None, None
        
        # 使用检测到的第一个人脸 (如果有多个人脸)
        face_location = face_locations[0]
        
        # 提取人脸编码
        face_encodings = face_recognition.face_encodings(rgb_image, [face_location])
        
        if not face_encodings:
            print(f"无法提取人脸特征: {image_path}")
            return None, None
        
        return face_encodings[0], face_location
    
    def extract_face_image(self, image_path, face_location):
        """
        从原始图像中提取人脸区域的图像
        
        参数:
            image_path: 图像文件路径
            face_location: 人脸位置坐标 (top, right, bottom, left)
            
        返回:
            face_image: 裁剪后的人脸图像的二进制数据
        """
        # 读取图像
        image = cv2.imread(image_path)
        
        # 提取坐标
        top, right, bottom, left = face_location
        
        # 裁剪人脸区域
        face_image = image[top:bottom, left:right]
        
        # 调整大小为统一尺寸 (可选)
        face_image = cv2.resize(face_image, (150, 150))
        
        # 转换为二进制数据
        _, face_bytes = cv2.imencode('.jpg', face_image)
        return face_bytes.tobytes()
    
    def process_images(self):
        """处理目录中的所有图像并将其向量化后存入Milvus"""
        # 确保集合存在
        self.create_collection()
        
        # 获取所有图像文件
        image_files = []
        for ext in ['jpg', 'jpeg', 'png']:
            image_files.extend(glob.glob(os.path.join(self.image_dir, f'*.{ext}')))
        
        if not image_files:
            print(f"在 {self.image_dir} 中未找到图像文件")
            return
        
        print(f"找到 {len(image_files)} 个图像文件，开始处理...")
        
        # 准备数据
        entities = []
        
        for image_path in tqdm(image_files):
            # 从文件名提取人名
            base_name = os.path.basename(image_path)
            name = os.path.splitext(base_name)[0]
            
            # 提取人脸编码和位置
            face_encoding, face_location = self.extract_face_encoding(image_path)
            
            if face_encoding is None:
                continue
            
            # 准备插入Milvus的实体
            entity = {
                NAME_FIELD_NAME: name,
                PATH_FIELD_NAME: image_path,
                EMBEDDING_FIELD_NAME: face_encoding.tolist()
            }
            
            entities.append(entity)
        
        if not entities:
            print("没有有效的人脸可以处理")
            return
        
        # 插入Milvus
        try:
            print("正在向Milvus插入数据...")
            result = self.client.insert(
                collection_name=COLLECTION_NAME,
                data=entities
            )
            
            print(f"成功插入 {len(entities)} 个人脸特征向量!")
            print(f"插入结果: {result}")
            
        except Exception as e:
            print(f"插入数据失败: {e}")
    
    def search_similar_faces(self, query_image_path, top_k=5):
        """
        在Milvus中搜索与查询图像相似的人脸
        
        参数:
            query_image_path: 查询图像路径
            top_k: 返回的最相似结果数量
            
        返回:
            results: 搜索结果列表
        """
        # 提取查询图像的人脸编码
        face_encoding, _ = self.extract_face_encoding(query_image_path)
        
        if face_encoding is None:
            print(f"无法从查询图像中提取人脸特征: {query_image_path}")
            return []
        
        # 执行向量搜索
        search_params = {"metric_type": "COSINE", "params": {"ef": 128}}
        
        results = self.client.search(
            collection_name=COLLECTION_NAME,
            data=[face_encoding.tolist()],
            anns_field=EMBEDDING_FIELD_NAME,
            param=search_params,
            limit=top_k,
            output_fields=[NAME_FIELD_NAME, PATH_FIELD_NAME]
        )
        
        return results

if __name__ == "__main__":
    vectorizer = FaceVectorizer()
    vectorizer.process_images()
    
    # 测试搜索功能 (可选)
    # 如果存在测试图像，可以取消下面注释进行测试
    # test_image = "path/to/test/image.jpg"
    # if os.path.exists(test_image):
    #     results = vectorizer.search_similar_faces(test_image)
    #     print(f"查询结果: {results}") 