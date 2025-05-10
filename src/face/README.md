# 人脸向量化与相似度搜索

这个项目实现了人脸图像的向量化处理和相似度搜索功能。使用了 face_recognition 库进行人脸检测和特征提取，并将结果存储在 Milvus 向量数据库中，以便进行高效的相似度搜索。

## 功能特性

- 自动检测图像中的人脸
- 提取人脸特征向量
- 将人脸图像和特征向量存储到 Milvus 向量数据库
- 基于向量相似度进行人脸搜索
- 可视化搜索结果

## 依赖项

- Python 3.6+
- OpenCV
- NumPy
- face_recognition
- pymilvus
- tqdm
- argparse

## 安装依赖

```bash
pip install face_recognition opencv-python numpy pymilvus tqdm argparse
```

## 使用方法

### 1. 准备人脸图像

将人脸图像放在`src/face/image`目录下。建议使用清晰的正面人脸照片，并以人名命名图像文件（例如：`张三.jpg`、`李四.png`等）。

### 2. 启动 Milvus 服务

确保 Milvus 服务已启动并运行在默认端口（19530）。如果使用不同配置，请在`face_vectorization.py`文件中修改相应的`MILVUS_HOST`和`MILVUS_PORT`常量。

### 3. 处理人脸图像

运行`face_vectorization.py`脚本，将处理`src/face/image`目录下的所有图像并存储到 Milvus 中：

```bash
python face_vectorization.py
```

### 4. 搜索相似人脸

使用`face_search_demo.py`脚本搜索与指定图像相似的人脸：

```bash
python face_search_demo.py --query path/to/query/image.jpg
```

其他可选参数：

- `--top-k`或`-k`：指定返回的最相似结果数量（默认为 5）
- `--display`或`-d`：显示结果图像（可视化）

例如：

```bash
python face_search_demo.py --query test.jpg --top-k 3 --display
```

## 文件说明

- `face_vectorization.py`：主要的人脸向量化处理模块
- `face_search_demo.py`：人脸搜索演示脚本
- `image/`：存放人脸图像的目录

## 工作原理

1. 使用`face_recognition`库检测图像中的人脸
2. 提取人脸的 128 维特征向量
3. 将特征向量和人脸图像数据存储到 Milvus 向量数据库
4. 使用余弦相似度进行向量搜索，找到最相似的人脸

## 注意事项

- 为获得更好的识别效果，建议使用清晰的正面人脸照片
- 图像文件名将被用作人名标识，请使用有意义的名称
- 该项目仅用于学习和研究目的
