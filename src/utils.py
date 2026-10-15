"""
utils.py - 工具函数模块
======================
提供配置加载、日志初始化、路径管理、运行管理等公共功能。
项目中所有其他模块都通过本模块获取配置和日志实例。
"""

import os
import json
import yaml
import logging
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Tuple


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
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


# ============================================================
# 运行管理：递增序号 + 时间戳双参数，每次运行独立输出
# ============================================================
_current_run_id: Optional[str] = None   # 如 "run0001_20261015_143022"
_RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
_COUNTER_FILE = _RUNS_DIR / "run_counter.json"


def convert_to_json_serializable(obj: Any) -> Any:
    """
    将对象转换为 JSON 可序列化的格式。

    处理 numpy 类型、DataFrame、Path 等常见非标准类型。
    """
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, pd.DataFrame):
        return {"__type__": "DataFrame", "shape": list(obj.shape), "columns": list(obj.columns)}
    elif isinstance(obj, pd.Series):
        return {"__type__": "Series", "length": len(obj), "name": obj.name}
    elif isinstance(obj, dict):
        return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return convert_to_json_serializable(obj.__dict__)
    else:
        try:
            if hasattr(obj, 'item'):
                return obj.item()
        except Exception:
            pass
        return str(obj)


def _load_counter() -> Tuple[int, list]:
    """加载运行计数器，返回 (next_id, records)"""
    if _COUNTER_FILE.exists():
        try:
            with open(_COUNTER_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("next_id", 1), data.get("records", [])
        except Exception:
            pass
    return 1, []


def _save_counter(next_id: int, records: list):
    """保存运行计数器"""
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    data = {"next_id": next_id, "records": records}
    tmp = _COUNTER_FILE.with_suffix(".json.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.rename(_COUNTER_FILE)
    except Exception:
        with open(_COUNTER_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def init_run(config: dict, run_id: Optional[int] = None) -> str:
    """
    初始化一次运行。

    分配递增序号，生成 run<NNNN>_<YYYYMMDD_HHMMSS> 格式的目录名，
    保存 config.json（含完整参数、时间戳、git commit）。

    参数:
        config: 配置字典
        run_id: 可选的手动指定序号。不传则自动递增。

    返回:
        当前 run_id 字符串（如 "run0001_20261015_143022"）
    """
    global _current_run_id

    next_id, records = _load_counter()

    if run_id is not None:
        num = run_id
    else:
        num = next_id

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"run{num:04d}_{timestamp}"

    _current_run_id = run_dir_name
    run_dir = get_run_dir()
    ensure_dir(run_dir)

    # 保存运行参数
    run_meta = {
        "run_id": num,
        "run_dir": run_dir_name,
        "timestamp": datetime.now().isoformat(),
        "config": convert_to_json_serializable(config),
    }

    # 尝试记录 git commit hash
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=5
        )
        if result.returncode == 0:
            run_meta["git_commit"] = result.stdout.strip()
    except Exception:
        pass

    config_path = run_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    # 更新计数器
    if run_id is None:
        next_id = num + 1
    else:
        next_id = max(next_id, num + 1)
    records.append({"run_id": num, "run_dir": run_dir_name, "timestamp": timestamp})
    _save_counter(next_id, records)

    return run_dir_name


def get_run_dir() -> Optional[Path]:
    """
    获取当前运行的输出目录。

    返回:
        run 目录的 Path 对象；如果未初始化 run 则返回 None
    """
    if _current_run_id is None:
        return None
    return PROJECT_ROOT / "outputs" / "runs" / _current_run_id


def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    创建并配置日志记录器。

    每个模块调用时传入自己的模块名，日志会统一格式输出到控制台。
    如果已初始化 run 目录，日志会同时写入 run.log 文件。

    参数:
        name: 日志记录器名称（通常传模块名）
        log_level: 日志级别，如 "DEBUG", "INFO", "WARNING"

    返回:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler（模块被多次 import 时）
    if not logger.handlers:
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # 控制台 handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 文件 handler（如果 run 目录已初始化）
        run_dir = get_run_dir()
        if run_dir is not None:
            log_file = run_dir / "run.log"
            file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

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

    如果已初始化 run，路径自动包含 run_id：outputs/runs/<run_id>/<subfolder>
    否则回退到原始路径：outputs/<subfolder>

    参数:
        config: 配置字典
        subfolder: "figures" 或 "results"

    返回:
        输出目录的 Path 对象
    """
    run_dir = get_run_dir()
    if run_dir is not None:
        return run_dir / subfolder
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
