# 人脸向量可视化项目

这个项目用于可视化展示人脸特征向量之间的相似关系。它通过从 Milvus 向量数据库中读取人脸特征向量，计算相似度，并以直观的网络图形式展示。

## 功能特点

- 从 Milvus 获取人脸向量数据
- 计算每个人脸之间的余弦相似度
- 将人脸图像作为节点展示在网络图中
- 通过连线表示人脸之间的相似关系（线越粗、颜色越亮表示相似度越高）
- 鼠标悬停在人脸上可以查看对应的向量数据
- 可调整相似度阈值筛选显示的连接
- 支持缩放、平移和拖拽操作的交互式可视化

## 项目结构

```
src/face/
├── face_vectorization.py  # 人脸向量化处理模块
├── face_api.py            # 后端API接口
├── main.py                # 应用入口
├── image/                 # 人脸图像目录
└── web/                   # 前端文件
    ├── index.html         # 主页面
    └── static/            # 静态资源
        ├── style.css      # 样式表
        └── graph.js       # 可视化逻辑
```

## 依赖要求

- Python 3.7+
- FastAPI
- Uvicorn
- NumPy
- OpenCV
- face_recognition
- PyMilvus
- 浏览器支持(推荐 Chrome, Firefox, Safari 最新版本)

## 安装与运行

1. **安装依赖**

```bash
pip install fastapi uvicorn numpy opencv-python face_recognition pymilvus
```

2. **向量化人脸图像**

首先需要运行人脸向量化脚本，提取人脸特征并存入 Milvus:

```bash
cd src/face
python face_vectorization.py
```

3. **启动可视化应用**

```bash
python main.py
```

应用将在 http://localhost:8000 启动。

## 使用方法

1. 访问 http://localhost:8000
2. 页面将显示所有人脸图像，以及它们之间的相似关系
3. 拖动滑块可以调整相似度阈值，筛选显示的连接
4. 鼠标悬停在人脸上可以查看详细的向量数据
5. 使用鼠标滚轮缩放，拖拽节点调整布局

## 注意事项

- 确保 Milvus 服务已经启动并运行在默认地址 (localhost:19530)
- 确保已经通过`face_vectorization.py`提前处理好了人脸图像并存入 Milvus
- 如需添加新的人脸图像，将图片放入`image`目录，然后重新运行`face_vectorization.py`
