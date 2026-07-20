import os
import shutil
import uuid
from typing import List, Dict, Any

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import func, select, case
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models.orange import OrangeTree
from app.schemas.orange import (
    SpatialQuerySchema,
    DiagnoseResultSchema,
    FertilizerStat,
    OrangeTreeOut,
)
from app.services.tif_service import TifResolver

router = APIRouter(prefix="/orange", tags=["脐橙三维空间大屏诊断API"])



@router.post(
    "/spatial-diagnose",
    response_model=DiagnoseResultSchema,
    status_code=status.HTTP_200_OK,
    summary="大屏鼠标拉框空间相交诊断接口",
)
async def spatial_diagnose(
    payload: SpatialQuerySchema,
    db: AsyncSession = Depends(get_session),
):
    """
    大屏拉框空间诊断接口：
    1. 接收前端 Cesium 传过来的 WGS84 闭合多边形
    2. PostGIS ST_Transform 把经纬度渔网投射到 UTM 32650
    3. GIST 空间索引秒级检索框内树木 + 聚合看板指标
    """
    coords = payload.coordinates

    # 智能识别经纬度顺序 + 自动强行闭合
    valid_coords = []
    for pt in coords:
        if len(pt) < 2:
            continue
        if pt[0] < pt[1]:  # pt[0]是纬度(≈27)，pt[1]是经度(≈116)，调换
            lng, lat = pt[1], pt[0]
        else:              # pt[0]是经度(≈116)，pt[1]是纬度(≈27)，不变
            lng, lat = pt[0], pt[1]
        valid_coords.append(f"{lng} {lat}")

    # 如果首尾不同，后端强制把口袋扎紧
    if valid_coords and valid_coords[0] != valid_coords[-1]:
        valid_coords.append(valid_coords[0])

    wkt_polygon = f"POLYGON(({', '.join(valid_coords)}))"

    polygon_4326 = func.ST_GeomFromText(wkt_polygon, 4326)
    target_roi = func.ST_Transform(polygon_4326, 32650)

    # 一次 I/O 聚合查询：计数 + 均值 + 施肥分级统计
    stmt = (
        select(
            func.count(OrangeTree.id).label("total_count"),
            func.avg(OrangeTree.height_m).label("avg_height"),
            func.avg(OrangeTree.shape_area).label("avg_area"),
            func.avg(OrangeTree.growth_index).label("avg_growth_index"),
            func.sum(case((OrangeTree.fertilizer_level == 1, 1), else_=0)).label("light_count"),
            func.sum(case((OrangeTree.fertilizer_level == 2, 1), else_=0)).label("medium_count"),
            func.sum(case((OrangeTree.fertilizer_level == 3, 1), else_=0)).label("heavy_count"),
        )
        .where(func.ST_Contains(target_roi, OrangeTree.geom))
    )

    result = (await db.execute(stmt)).one_or_none()

    if result is None or result.total_count == 0:
        return DiagnoseResultSchema(
            total_count=0,
            avg_height=0.0,
            avg_area=0.0,
            avg_growth_index=0.0,
            fertilizer_recommendation=FertilizerStat(),
            trees=[],
        )

    # 查询框内树木明细：ST_Transform 转回 WGS84，ST_X/ST_Y 现场拆解为纯数字经纬度
    tree_stmt = (
        select(
            OrangeTree.id,
            OrangeTree.batch_id,
            OrangeTree.confidence,
            OrangeTree.compactness,
            OrangeTree.shape_length,
            OrangeTree.shape_area,
            OrangeTree.value_field,
            OrangeTree.count_field,
            OrangeTree.area_m2,
            OrangeTree.height_m,
            OrangeTree.crown_diameter,
            OrangeTree.volume_m3,
            OrangeTree.growth_index,
            OrangeTree.slope_degree,
            OrangeTree.aspect,
            OrangeTree.fertilizer_level,
            func.ST_X(func.ST_Transform(OrangeTree.geom, 4326)).label("lng"),
            func.ST_Y(func.ST_Transform(OrangeTree.geom, 4326)).label("lat"),
        )
        .where(func.ST_Contains(target_roi, OrangeTree.geom))
        .limit(500)
    )
    tree_rows = (await db.execute(tree_stmt)).all()

    trees_out = []
    for row in tree_rows:
        trees_out.append(OrangeTreeOut(
            id=row.id,
            batch_id=row.batch_id,
            lng=round(row.lng, 6),
            lat=round(row.lat, 6),
            confidence=row.confidence,
            compactness=row.compactness,
            shape_length=row.shape_length,
            shape_area=row.shape_area,
            value_field=row.value_field,
            count_field=row.count_field,
            area_m2=row.area_m2,
            height_m=row.height_m,
            crown_diameter=row.crown_diameter,
            volume_m3=row.volume_m3,
            growth_index=row.growth_index,
            slope_degree=row.slope_degree,
            aspect=row.aspect,
            fertilizer_level=row.fertilizer_level,
        ))

    return DiagnoseResultSchema(
        total_count=result.total_count,
        avg_height=round(result.avg_height, 2) if result.avg_height else None,
        avg_area=round(result.avg_area, 2) if result.avg_area else None,
        avg_growth_index=round(result.avg_growth_index, 4) if result.avg_growth_index else None,
        fertilizer_recommendation=FertilizerStat(
            light_level_count=result.light_count or 0,
            medium_level_count=result.medium_count or 0,
            heavy_level_count=result.heavy_count or 0,
        ),
        trees=trees_out,
    )

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class FreshTreeResponse(BaseModel):
    tree_id: int
    geometry_geojson: Dict[str, Any]
    attributes: Dict[str, Any]

class TifUploadResponse(BaseModel):
    success: bool
    message: str
    file_path: str
    spatial_info: Dict[str, Any]  # 3.2 新增：回传前端核验的空间参考信息
    fresh_trees: List[FreshTreeResponse]


@router.post("/upload-tif", response_model=TifUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_orange_tif(file: UploadFile = File(...)):
    """
    接口 B: 接收前端上传的最新无人机正射二进制 TIF 文件并安全落地 + 现场空间参考扣留
    """
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in [".tif", ".tiff"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法文件格式！系统只接受 .tif 或 .tiff 格式的无人机正射影像。"
        )
        
    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
    target_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    try:
        # 3.1 阶段：流式对拷物理落地
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        if os.path.exists(target_path):
            os.remove(target_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文件流写入服务器硬盘失败: {str(e)}"
        )
    finally:
        await file.close()

    # 获取落地的绝对物理路径
    absolute_file_path = os.path.abspath(target_path)
    
    # ==================== 【3.2 关键合流】：大总台实例化并提取空间参考 ====================
    try:
        # 1. 实例化方法类，交接物理文件路径的接力棒
        resolver = TifResolver(file_path=absolute_file_path)
        
        # 2. 调用核心方法，现场去拔大文件的地理皮肤
        spatial_data = resolver.extract_spatial_reference()
        
    except Exception as geo_err:
        # 如果 rasterio 在读取地理头信息时崩溃（如 TIF 损坏或 conda 依赖损坏），及时报错
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"【3.2 空间参考扣留失败】请检查 Conda 环境中的 rasterio 依赖: {str(geo_err)}"
        )
    # ==================================================================================

    # 【当前战术妥协】：下游算法继续用 Mock 数据垫付
    mock_fresh_trees = [
        {
            "tree_id": 999,
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
                "fertilizer_level": 2
            }
        }
    ]

    # 返回标准的响应，里面带上了刚由类抓出来的真实物理影像坐标数据
    return TifUploadResponse(
        success=True,
        message="【3.1+3.2 双重通关】无人机影像接收成功且空间参考提取完毕！",
        file_path=absolute_file_path,
        spatial_info=spatial_data, # 吐给前端看，证明后端真的把照片底细摸清了
        fresh_trees=mock_fresh_trees
    )