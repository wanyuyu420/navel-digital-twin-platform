import os
import shutil
import uuid
import math
import time
from typing import List, Dict, Any

import httpx
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import func, select, case
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_session
from app.models.orange import OrangeTree
from app.schemas.orange import (
    SpatialQuerySchema,
    DiagnoseResultSchema,
    FertilizerStat,
    OrangeTreeOut,
    HistoricalTreesOut,
    TreeFeature,
    growth_index_to_status,
    fertilizer_level_to_kg,
)
from app.services.tif_service import TifService
from pyproj import Transformer
from app.services.sam_service import SamInferenceService
from app.services.yolo_service import YoloService
import rasterio
import cv2
import numpy as np

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


@router.get(
    "/historical-trees",
    response_model=HistoricalTreesOut,
    status_code=status.HTTP_200_OK,
    summary="开屏静态会师 — 历史老树全量坐标与属性",
)
async def get_historical_trees(db: AsyncSession = Depends(get_session)):
    """
    大屏开屏时前端一次性拉取全部历史老树（batch_id=historical_zone）的
    经纬度坐标与长势/施肥属性，用于在 3D 底图模型表面铺设隐形拾取点。
    """
    stmt = (
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
        .where(OrangeTree.batch_id == "historical_zone")
    )
    rows = (await db.execute(stmt)).all()

    trees_out = []
    for row in rows:
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

    return HistoricalTreesOut(total=len(trees_out), trees=trees_out)


UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class TifUploadResponse(BaseModel):
    success: bool
    message: str
    file_path: str
    spatial_info: Dict[str, Any]
    task_id: str = ""


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
        # 使用 rasterio 直接提取空间参考（TifService 替代旧 TifResolver）
        
        # 2. 利用 rasterio 直接读取空间参考
        import rasterio as _rio
        with _rio.open(absolute_file_path) as _src:
            spatial_data = {
                "crs": str(_src.crs),
                "transform": [t for t in _src.transform] if _src.transform else [],
            }
        
    except Exception as geo_err:
        # 如果 rasterio 在读取地理头信息时崩溃（如 TIF 损坏或 conda 依赖损坏），及时报错
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"【3.2 空间参考扣留失败】请检查 Conda 环境中的 rasterio 依赖: {str(geo_err)}"
        )
    # ==================================================================================

    # Start real YOLO+SAM inference task
    task_id = uuid.uuid4().hex[:12]
    with _task_lock:
        _task_store[task_id] = {
            "task_id": task_id,
            "status": "pending",
            "message": "Task created",
            "total_trees": 0,
            "fresh_trees": [],
            "progress": 0.0,
        }

    asyncio.create_task(asyncio.to_thread(_run_inference_task, task_id, absolute_file_path))

    return TifUploadResponse(
        success=True,
        message="Upload success, YOLO+SAM inference started in background",
        file_path=absolute_file_path,
        spatial_info=spatial_data,
        task_id=task_id,
    )
import asyncio
import threading
from datetime import datetime


# ===== Async task store =====

_task_store: dict = {}
_task_lock = threading.Lock()


class TaskStatusOut(BaseModel):
    task_id: str
    status: str  # pending | processing | completed | failed
    message: str = ""
    total_trees: int = 0
    fresh_trees: list = []
    progress: float = 0.0  # 0.0 ~ 1.0


def _calc_growth_fields(mask: np.ndarray, gsd: float, height_m: float = None) -> dict:
    """Calculate canopy growth fields from SAM segmentation mask and optional height."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"area_m2": 0, "crown_diameter": 0, "shape_length": 0,
                "compactness": 0, "volume_m3": None, "height_m": None,
                "growth_index": None, "fertilizer_level": 0}

    area_px = int(cv2.contourArea(contours[0]))
    perimeter_px = cv2.arcLength(contours[0], closed=True)
    if perimeter_px == 0:
        return {"area_m2": 0, "crown_diameter": 0, "shape_length": 0,
                "compactness": 0, "volume_m3": None, "height_m": None,
                "growth_index": None, "fertilizer_level": 0}

    area_m2 = area_px * gsd * gsd
    shape_length = perimeter_px * gsd
    crown_diameter = 2.0 * math.sqrt(area_m2 / math.pi)
    compactness = (4.0 * math.pi * area_px) / (perimeter_px * perimeter_px)

    # Volume: cone approximation (area * height / 3)
    volume_m3 = (area_m2 * height_m / 3.0) if height_m and height_m > 0 else None

    # Growth index: composite 0-1 score from compactness + height-to-crown ratio
    if height_m and height_m > 0 and crown_diameter > 0:
        hc_ratio = height_m / crown_diameter
        hc_score = max(0.0, 1.0 - abs(hc_ratio - 1.0))
        growth_index = round(compactness * 0.5 + hc_score * 0.5, 4)
    else:
        growth_index = round(compactness, 4)

    # Fertilizer recommendation based on growth index
    if growth_index >= 0.7:
        fertilizer_level = 1   # light
    elif growth_index >= 0.4:
        fertilizer_level = 2   # medium
    else:
        fertilizer_level = 3   # heavy

    return {
        "area_pixels": area_px,
        "area_m2": round(area_m2, 4),
        "shape_length": round(shape_length, 4),
        "crown_diameter": round(crown_diameter, 4),
        "compactness": round(compactness, 4),
        "height_m": round(height_m, 2) if height_m and height_m > 0 else None,
        "volume_m3": round(volume_m3, 4) if volume_m3 else None,
        "growth_index": growth_index,
        "fertilizer_level": fertilizer_level,
    }


# ===== GeoScene Server 集成 =====

_geoscene_token: dict = {}


def _get_geoscene_token() -> str | None:
    """获取 GeoScene Server Token，带缓存"""
    settings = get_settings()
    if not settings.geoscene_server_url or not settings.geoscene_username:
        return None

    now = time.time()
    cached = _geoscene_token.get("token")
    expires = _geoscene_token.get("expires", 0)
    if cached and expires > now + 60:
        return cached

    try:
        resp = httpx.post(
            f"{settings.geoscene_server_url}/tokens/generateToken",
            data={
                "username": settings.geoscene_username,
                "password": settings.geoscene_password,
                "client": "referer",
                "referer": settings.geoscene_server_url,
                "expiration": settings.geoscene_token_duration,
                "f": "json",
            },
            timeout=10,
        )
        data = resp.json()
        _geoscene_token["token"] = data["token"]
        _geoscene_token["expires"] = now + settings.geoscene_token_duration * 60
        return _geoscene_token["token"]
    except Exception as e:
        print(f"[GeoScene] Token fetch failed: {e}")
        return None


def _publish_to_geoscene(trees_data: list, batch_id: str):
    """将推理检测到的树木同步发布到 GeoScene FeatureServer"""
    token = _get_geoscene_token()
    if not token:
        print("[GeoScene] Skipping publish - no valid token")
        return

    settings = get_settings()
    if not settings.geoscene_feature_server_url:
        print("[GeoScene] Skipping publish - no feature server URL configured")
        return

    adds = []
    for tree in trees_data:
        adds.append({
            "geometry": {
                "x": tree["utm_x"],
                "y": tree["utm_y"],
                "spatialReference": {"wkid": 32650},
            },
            "attributes": {
                "batch_id": batch_id,
                "confidence": tree.get("iou_score"),
                "compactness": tree.get("compactness"),
                "shape_length": tree.get("shape_length"),
                "shape_area": tree.get("area_m2"),
                "area_m2": tree.get("area_m2"),
                "height_m": tree.get("height_m"),
                "crown_diameter": tree.get("crown_diameter"),
                "volume_m3": tree.get("volume_m3"),
                "growth_index": tree.get("growth_index"),
                "slope_degree": tree.get("slope_degree"),
                "aspect": tree.get("aspect"),
                "fertilizer_level": tree.get("fertilizer_level", 0),
            },
        })

    try:
        url = f"{settings.geoscene_feature_server_url}/0/applyEdits"
        resp = httpx.post(
            url,
            params={"f": "json", "token": token},
            json={"adds": adds},
            timeout=60,
        )
        result = resp.json()
        if result.get("addResults"):
            success = sum(
                1 for r in result["addResults"] if r.get("success")
            )
            print(f"[GeoScene] Published {success}/{len(adds)} trees to FeatureServer")
        else:
            err = result.get("error", resp.text)
            print(f"[GeoScene] Publish error: {err}")
    except Exception as e:
        print(f"[GeoScene] Publish failed: {e}")


def _persist_trees_sync(trees_data: list, batch_id: str):
    """Persist detected trees to DB via sync connection (runs in background thread)."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    if not trees_data:
        return

    settings = get_settings()
    engine = create_engine(settings.database_url_sync, echo=False)
    try:
        with Session(engine) as session:
            for tree in trees_data:
                session.add(OrangeTree(
                    batch_id=batch_id,
                    geom=f"POINT({tree['utm_x']} {tree['utm_y']})",
                    confidence=tree.get("iou_score"),
                    compactness=tree.get("compactness"),
                    shape_length=tree.get("shape_length"),
                    shape_area=tree.get("shape_area"),
                    value_field=tree.get("value"),
                    area_m2=tree.get("area_m2"),
                    height_m=tree.get("height_m"),
                    crown_diameter=tree.get("crown_diameter"),
                    volume_m3=tree.get("volume_m3"),
                    growth_index=tree.get("growth_index"),
                    slope_degree=tree.get("slope_degree"),
                    aspect=tree.get("aspect"),
                    fertilizer_level=tree.get("fertilizer_level", 0),
                ))
            session.commit()
            print(f"[Persist] {len(trees_data)} trees saved to DB (batch: {batch_id})")
    except Exception as e:
        print(f"[Persist] Failed to save trees: {e}")
    finally:
        engine.dispose()


def _make_geojson_from_mask(
    mask: np.ndarray,
    window_x: int,
    window_y: int,
    transform,
    wgs84_transformer,
) -> dict | None:
    """
    将SAM输出的二进制分割mask转为WGS84 GeoJSON Polygon。
    抽取最大外轮廓 → 简化顶点 → 像素坐标转WGS84经纬度 → 闭合环。
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest, closed=True)
    if perimeter < 4.0:
        return None

    epsilon = 0.005 * perimeter  # 保留 99.5% 周长精度，大幅压缩顶点数
    simplified = cv2.approxPolyDP(largest, epsilon, closed=True)

    coords = []
    for pt in simplified:
        px, py = pt[0]
        global_px = window_x + float(px)
        global_py = window_y + float(py)
        geo_x, geo_y = rasterio.transform.xy(transform, global_py, global_px)
        lng, lat = wgs84_transformer.transform(geo_x, geo_y)
        coords.append([round(lng, 8), round(lat, 8)])

    if len(coords) < 4:
        return None

    # GeoJSON 要求首尾坐标一致
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    return {"type": "Polygon", "coordinates": [coords]}


def _run_inference_task(task_id: str, file_path: str):
    with _task_lock:
        _task_store[task_id]["status"] = "processing"

    try:
        yolo_model = YoloService.get_instance()
        sam_predictor = SamInferenceService.get_instance()

        import rasterio as _rio
        with _rio.open(file_path) as _src:
            tif_crs = str(_src.crs)
            gsd = float(_src.res[0])  # meters per pixel

        transformer = Transformer.from_crs(tif_crs, "EPSG:4326", always_xy=True)
        transformer_utm = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)

        batch_id = os.path.splitext(os.path.basename(file_path))[0]

        # Step 0: Predict full canopy height map from RGB
        _update_progress(task_id, 0, 1, "Predicting canopy height...")
        try:
            from app.services.height_service import HeightService
            from app.services.elevation_service import ElevationService
            height_map = HeightService.predict_height_map(file_path)
        except Exception as e:
            print(f"[Warning] Height prediction failed: {e}, skipping height fields")
            height_map = None

        all_detected_trees = []
        tile_count = 0
        rio_ds = rasterio.open(file_path)

        tiles = list(TifService.slice_tif_generator(file_path))
        total_tiles = len(tiles)

        for tile_info in tiles:
            tile_rgb = tile_info["tile_data"]
            valid_mask = tile_info["valid_mask"]
            window_x = tile_info["window_x"]
            window_y = tile_info["window_y"]
            transform = tile_info["transform"]

            # Smart skip: blank or low-contrast tiles
            if tile_rgb.max() < 10 or tile_rgb.std() < 5:
                tile_count += 1
                _update_progress(task_id, tile_count, total_tiles)
                continue

            # Step 1: YOLO detects tree canopy boxes
            boxes = YoloService.detect_boxes(tile_rgb, yolo_model, conf=0.15)
            if len(boxes) == 0:
                tile_count += 1
                _update_progress(task_id, tile_count, total_tiles)
                continue

            # Step 2: SAM refines each box with Box Prompt
            local_trees = SamInferenceService.infer_tile_with_boxes(
                tile_rgb, valid_mask, boxes, sam_predictor)

            # Step 3: Coordinate conversion + post-processing growth fields
            for tree in local_trees:
                local_cx, local_cy = tree["local_centroid"]
                global_px = window_x + local_cx
                global_py = window_y + local_cy
                geo_x, geo_y = rasterio.transform.xy(
                    transform, global_py, global_px, offset="center")
                lng, lat = transformer.transform(geo_x, geo_y)
                bbox = tree.get("bbox", (0, 0, 0, 0))
                tree_h = float(np.median(height_map[max(0,int(global_py)-2):min(height_map.shape[0],int(global_py)+3), max(0,int(global_px)-2):min(height_map.shape[1],int(global_px)+3)][height_map[max(0,int(global_py)-2):min(height_map.shape[0],int(global_py)+3), max(0,int(global_px)-2):min(height_map.shape[1],int(global_px)+3)] > 0])) if height_map is not None and 0 <= int(global_py) < height_map.shape[0] and 0 <= int(global_px) < height_map.shape[1] else None
                growth = _calc_growth_fields(tree["segmentation_mask"], gsd, tree_h)
                utm_x, utm_y = transformer_utm.transform(lng, lat)
                slope_info = ElevationService.get_slope_aspect(lat, lng, utm_x, utm_y); band_val = tile_rgb[int(local_cy), int(local_cx)].tolist(); raw_val = float(rio_ds.read(1, window=((int(global_py),int(global_py)+1), (int(global_px),int(global_px)+1)))[0,0]) if 0 <= int(global_py) < rio_ds.shape[0] and 0 <= int(global_px) < rio_ds.shape[1] else None if 0 <= int(local_cy) < tile_rgb.shape[0] and 0 <= int(local_cx) < tile_rgb.shape[1] else None
                geojson = _make_geojson_from_mask(
                    tree["segmentation_mask"], window_x, window_y,
                    transform, transformer)
                tree_uuid = f"tree_{uuid.uuid4().hex[:8]}"
                all_detected_trees.append({
                    "id": tree_uuid,
                    "batch_id": batch_id,
                    "lng": round(lng, 8),
                    "lat": round(lat, 8),
                    "utm_x": round(utm_x, 4),
                    "utm_y": round(utm_y, 4),
                    "iou_score": round(tree.get("iou_score", 0), 4),
                    "bbox_local": [round(float(v), 2) for v in bbox],
                    "shape_area": growth.get("area_m2", 0),
                    "band_value": band_val, "value": raw_val, **slope_info,
                    **growth,
                    "growth_status": growth_index_to_status(growth.get("growth_index")),
                    "fertilizer_kg": fertilizer_level_to_kg(growth.get("fertilizer_level", 0)),
                    "geometry": geojson,
                })

            tile_count += 1
            _update_progress(task_id, tile_count, total_tiles)

        # Persist detected trees to database for spatial-diagnose
        _persist_trees_sync(all_detected_trees, batch_id)

        # Publish to GeoScene FeatureServer
        _publish_to_geoscene(all_detected_trees, batch_id)

        with _task_lock:
            _task_store[task_id]["status"] = "completed"
            _task_store[task_id]["total_trees"] = len(all_detected_trees)
            _task_store[task_id]["fresh_trees"] = all_detected_trees

        rio_ds.close()
        if os.path.exists(file_path):
            os.remove(file_path)

    except Exception as e:
        with _task_lock:
            _task_store[task_id]["status"] = "failed"
            _task_store[task_id]["message"] = str(e)
        rio_ds.close()
        if os.path.exists(file_path):
            os.remove(file_path)


def _update_progress(task_id, tile_count, total_tiles, msg: str = None):
    progress = tile_count / max(total_tiles, 1)
    with _task_lock:
        _task_store[task_id]["progress"] = progress
        _task_store[task_id]["message"] = msg if msg else f"{tile_count}/{total_tiles} tiles"


@router.post("/upload-and-interpret", response_model=TaskStatusOut)
async def upload_and_interpret_tif(file: UploadFile = File(...)):
    temp_dir = "temp_storage"
    os.makedirs(temp_dir, exist_ok=True)
    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    temp_file_path = os.path.join(temp_dir, unique_name)

    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    task_id = uuid.uuid4().hex[:12]
    with _task_lock:
        _task_store[task_id] = {
            "task_id": task_id,
            "status": "pending",
            "message": "Task created",
            "total_trees": 0,
            "fresh_trees": [],
            "progress": 0.0,
        }

    asyncio.create_task(asyncio.to_thread(_run_inference_task, task_id, temp_file_path))

    return {
        "task_id": task_id,
        "status": "pending",
        "message": "Task created, processing in background",
        "total_trees": 0,
        "fresh_trees": [],
        "progress": 0.0,
    }


@router.get("/upload-and-interpret/{task_id}", response_model=TaskStatusOut)
async def get_interpret_task(task_id: str):
    with _task_lock:
        task = _task_store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task