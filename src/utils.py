"""
utils.py - 工具函数模块
======================
提供配置加载、日志初始化、路径管理等公共功能。
项目中所有其他模块都通过本模块获取配置和日志实例。
"""

import os
import yaml
import logging
import pandas as pd
from pathlib import Path
from typing import Optional


# ============================================================
# 项目根目录：自动定位到 config/ 的上一级
# ============================================================
# 这样无论从哪里运行脚本，都能正确找到项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: Optional[str] = None) -> dict:
    """
    加载 YAML 配置文件。

    参数:
        config_path: 配置文件路径。如果不传，默认使用 config/config.yaml

    返回:
        解析后的字典，包含所有项目参数

    示例:
        >>> cfg = load_config()
        >>> cfg['data']['start_date']
        '2018-01-01'
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    创建并配置日志记录器。

    每个模块调用时传入自己的模块名，日志会统一格式输出到控制台。

    参数:
        name: 日志记录器名称（通常传 __name__）
        log_level: 日志级别，如 "DEBUG", "INFO", "WARNING"

    返回:
        配置好的 Logger 实例

    示例:
        >>> logger = setup_logger("data_loader")
        >>> logger.info("数据加载完成")
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler（模块被多次 import 时）
    if not logger.handlers:
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # 控制台输出格式：时间 | 模块名 | 级别 | 消息
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_data_path(config: dict, subfolder: str = "raw") -> Path:
    """
    获取数据目录的绝对路径。

    参数:
        config: 配置字典
        subfolder: "raw" 或 "processed"

    返回:
        数据目录的 Path 对象

    示例:
        >>> get_data_path(cfg, "raw")
        PosixPath('/path/to/project/data/raw')
    """
    return PROJECT_ROOT / "data" / subfolder


def get_output_path(config: dict, subfolder: str = "results") -> Path:
    """
    获取输出目录的绝对路径。

    参数:
        config: 配置字典
        subfolder: "figures" 或 "results"

    返回:
        输出目录的 Path 对象
    """
    return PROJECT_ROOT / "outputs" / subfolder


def ensure_dir(path: Path) -> Path:
    """
    确保目录存在，不存在则自动创建。

    参数:
        path: 目录路径

    返回:
        传入的路径（方便链式调用）
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(df: pd.DataFrame, filepath: Path, logger: logging.Logger = None):
    """
    将 DataFrame 保存为 parquet 文件。

    parquet 格式比 CSV 快很多，而且保留数据类型（日期、数值精度等）。

    参数:
        df: 要保存的 DataFrame
        filepath: 保存路径（需以 .parquet 结尾）
        logger: 可选的日志记录器
    """
    ensure_dir(filepath.parent)
    df.to_parquet(filepath, engine="pyarrow", index=True)
    if logger:
        logger.info(f"数据已保存: {filepath}，共 {len(df)} 行")


def load_dataframe(filepath: Path, logger: logging.Logger = None) -> pd.DataFrame:
    """
    从 parquet 文件加载 DataFrame。

    参数:
        filepath: parquet 文件路径
        logger: 可选的日志记录器

    返回:
        加载的 DataFrame
    """
    df = pd.read_parquet(filepath, engine="pyarrow")
    if logger:
        logger.info(f"数据已加载: {filepath}，共 {len(df)} 行")
    return df
