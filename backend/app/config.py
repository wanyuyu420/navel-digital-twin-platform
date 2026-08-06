from functools import lru_cache
from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    """Application configuration loaded from environment or .env"""

    # Default to SQLite for demo mode (no external DB required)
    # Set DATABASE_URL env var to use PostgreSQL in production
    database_url: str = "sqlite+aiosqlite:///./demo.db"
    database_url_sync: str = "sqlite:///./demo.db"

    # PostgreSQL example (set in .env for production):
    # database_url: str = "postgresql+asyncpg://user:password@localhost:5432/water_twin"
    # database_url_sync: str = "postgresql+psycopg2://user:password@localhost:5432/water_twin"

    api_prefix: str = "/api"
    enable_seed_data: bool = True  # Auto-seed demo data on startup
    debug: bool = True

    # GIS data directory - platform-specific absolute path
    # Windows:  D:\para\data\gis-data
    # macOS:    ~/data/gis-data
    # Ubuntu:   /home/user/data/gis-data
    gis_data_dir: str | None = None

    # ===== GeoScene Server 连接配置 =====
    geoscene_server_url: str = ""
    """GeoScene Server 根地址"""

    geoscene_feature_server_url: str = ""
    """要素服务地址"""

    geoscene_image_server_url: str = ""
    """影像服务地址"""

    geoscene_username: str = ""
    """GeoScene Server 用户名"""

    geoscene_password: str = ""
    """GeoScene Server 密码"""

    geoscene_token_duration: int = 120
    """Token 有效期（分钟）"""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def is_sqlite(self) -> bool:
        """Check if using SQLite database"""
        return "sqlite" in self.database_url.lower()


@lru_cache
def get_settings() -> Settings:
    return Settings()
