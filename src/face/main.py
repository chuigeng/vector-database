#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
人脸向量可视化应用入口
启动FastAPI服务器，提供API接口和静态文件服务
"""

import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# 导入API模块
from face_api import app as api_app

# 创建主应用
app = FastAPI(title="人脸向量可视化应用")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建web目录（如不存在）
os.makedirs("src/face/web", exist_ok=True)
os.makedirs("src/face/web/static", exist_ok=True)

# 挂载API路由
app.mount("/api", api_app)

# 设置静态文件
app.mount("/static", StaticFiles(directory="src/face/web/static"), name="static")

# 设置模板
templates = Jinja2Templates(directory="src/face/web")

@app.get("/")
async def read_root(request: Request):
    """返回主页面"""
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100) 