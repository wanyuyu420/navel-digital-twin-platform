# 1. 拉框空间诊断接口：接收坐标 $\rightarrow$ 范围检索 $\rightarrow$ 看板指标现场掐算 $\rightarrow$ 返回 JSON。
# 2. TIF上传解译接口：接收TIF $\rightarrow$ 扣留空间参考 $\rightarrow$ 原图滑动窗口切片喂给大模型 $\rightarrow$ 
# 后端 Area 和紧凑度清洗杂质 $\rightarrow$ 批量写入 PostgreSQL 数据库 $\rightarrow$ 返回 JSON 落图。
import os
import shutil
import uuid
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict, Any

router = APIRouter()

# 确保存放无人机影像的物理隔离文件夹存在
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 定义动态单体化要素回传的响应盒结构（符合 3.6 规范）
class FreshTreeResponse(BaseModel):
    tree_id: int
    geometry_geojson: Dict[str, Any]
    attributes: Dict[str, Any]

class TifUploadResponse(BaseModel):
    success: bool
    message: str
    file_path: str
    fresh_trees: List[FreshTreeResponse]


@router.post("/upload-tif", response_model=TifUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_orange_tif(file: UploadFile = File(...)):
    """
    接口 B: 接收前端上传的最新无人机正射二进制 TIF 文件并安全落地 (方案 A)
    """
    # 1. 安全检查：拦截非 TIF 格式文件，防止后续 rasterio 解析崩溃
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in [".tif", ".tiff"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法文件格式！系统只接受 .tif 或 .tiff 格式的无人机正射影像。"
        )
        
    # 2. 唯一命名：结合 UUID 或是时间戳，防止多人同时上传同名文件造成覆盖冲突
    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
    target_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    try:
        # 3. 流式对拷（方案 A 核心）：以缓冲区形式流式写入硬盘，16G 内存绝对不会爆仓
        with open(target_path, "wb") as buffer:
            # file.file 是 aiofiles 的 SpooledTemporaryFile 流对象
            shutil.copyfileobj(file.file, buffer)
            
    except Exception as e:
        # 如果中途发生写入异常，安全清理未写完的残渣文件，并抛出错误
        if os.path.exists(target_path):
            os.remove(target_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文件流写入服务器硬盘失败: {str(e)}"
        )
    finally:
        # 释放前端网络请求的文件资源
        await file.close()

    # 4. 获取落地的绝对物理路径，为明天装入 TifResolver 方法类做接力准备
    absolute_file_path = os.path.abspath(target_path)
    
    # =========================================================================
    # 【当前战术妥协】：下游算法服务未打通，先用 Mock（模拟）数据垫付，确保全栈河流畅通
    # =========================================================================
    mock_fresh_trees = [
        {
            "tree_id": 999,  # 模拟新生成的临时单体化ID
            "geometry_geojson": {
                "type": "Polygon",
                "coordinates": [[
                    [114.0, 25.0], 
                    [114.001, 25.0], 
                    [114.001, 25.001], 
                    [114.0, 25.001], 
                    [114.0, 25.0]
                ]]
            },
            "attributes": {
                "height_m": 2.8,
                "Area_m2": 4.1,
                "compactness": 0.88,
                "vari": 0.35,
                "fertilizer_level": 2  # 施肥等级先随机/固定给一个值
            }
        }
    ]
    # =========================================================================

    # 5. 打包标准响应盒回传前端
    return TifUploadResponse(
        success=True,
        message="无人机影像接收成功，方案 A 硬盘物理落地已完成，准备触发 GeoAI 级联解译。",
        file_path=absolute_file_path,
        fresh_trees=mock_fresh_trees
    )