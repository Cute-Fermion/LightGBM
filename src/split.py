"""
split.py - 时间序列滚动划分模块
===================================
将数据按时间划分为多个训练/验证/测试窗口，模拟真实的滚动预测场景。

为什么不能用随机划分（如 K-Fold）？
    金融数据有时间序列依赖性：未来的数据跟过去相关。
    如果随机划分，训练集和验证集可能包含相邻日期的数据，
    导致信息泄露（模型"看到"了跟验证集高度相关的训练样本）。

    正确做法：训练集必须在验证集之前，且中间留 embargo 间隔。

划分方式:
    滚动窗口（Rolling Window）：
    - 训练集固定长度（如 3 年 = 756 个交易日）
    - 验证集为接下来的 1 个月（20 个交易日）
    - 每月向前滚动一次
    - 总共生成多个 (train, valid) 对

示意图:
    Window 1:  [====train====][valid]
    Window 2:       [====train====][valid]
    Window 3:            [====train====][valid]
    ...

运行方式:
    python src/split.py --config config/config.yaml
"""

import argparse
import numpy as np
import pandas as pd
from typing import List, Tuple
from dataclasses import dataclass

from utils import (
    load_config, setup_logger, PROJECT_ROOT,
    get_data_path, save_dataframe, load_dataframe
)


@dataclass
class TimeWindow:
    """
    一个时间窗口的定义。

    属性:
        window_id: 窗口编号（从 0 开始）
        train_start: 训练集起始日期
        train_end: 训练集结束日期（不含）
        valid_start: 验证集起始日期
        valid_end: 验证集结束日期（不含）
        embargo_start: embargo 区间起始（训练集结束后）
        embargo_end: embargo 区间结束（验证集开始前）
    """
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    embargo_days: int = 5  # 默认 5 天 embargo


def generate_rolling_windows(dates: pd.DatetimeIndex, config: dict, logger) -> List[TimeWindow]:
    """
    生成滚动窗口列表。

    参数:
        dates: 数据中的所有交易日（排序后）
        config: 配置字典
        logger: 日志记录器

    返回:
        TimeWindow 列表
    """
    train_window = config["split"]["train_window"]      # 756（约3年）
    predict_window = config["split"]["predict_window"]   # 20（约1个月）
    step = config["split"]["step"]                       # 20（每月滚动）

    # embargo 天数：至少为前瞻收益的天数，防止标签重叠
    # 如果标签是 20 日前瞻收益，那训练集最后 20 天的标签会延伸到验证集
    # 需要把这些天的训练样本剔除
    forward_days = config["labels"]["forward_days"]
    embargo_days = forward_days

    logger.info(f"滚动窗口参数: 训练={train_window}日, 预测={predict_window}日, 步长={step}日")
    logger.info(f"Embargo: {embargo_days}日（防止标签重叠）")

    windows = []
    window_id = 0

    # 第一个窗口的训练集起始位置
    start_idx = 0

    while True:
        # 训练集范围
        train_start_idx = start_idx
        train_end_idx = start_idx + train_window

        # 验证集范围（训练集结束后，跳过 embargo）
        valid_start_idx = train_end_idx + embargo_days
        valid_end_idx = valid_start_idx + predict_window

        # 检查是否超出数据范围
        if valid_end_idx > len(dates):
            break

        window = TimeWindow(
            window_id=window_id,
            train_start=dates[train_start_idx],
            train_end=dates[train_end_idx - 1],
            valid_start=dates[valid_start_idx],
            valid_end=dates[min(valid_end_idx - 1, len(dates) - 1)],
            embargo_days=embargo_days
        )
        windows.append(window)

        # 向前滚动
        start_idx += step
        window_id += 1

    logger.info(f"生成 {len(windows)} 个滚动窗口")
    if windows:
        logger.info(f"  第一个窗口: 训练 {windows[0].train_start.date()} ~ {windows[0].train_end.date()}, "
                     f"验证 {windows[0].valid_start.date()} ~ {windows[0].valid_end.date()}")
        logger.info(f"  最后一个窗口: 训练 {windows[-1].train_start.date()} ~ {windows[-1].train_end.date()}, "
                     f"验证 {windows[-1].valid_start.date()} ~ {windows[-1].valid_end.date()}")

    return windows


def split_data_by_window(
    factors: pd.DataFrame,
    labels: pd.DataFrame,
    window: TimeWindow,
    logger
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按单个时间窗口切分数据。

    参数:
        factors: 预处理后的因子数据
        labels: 标签数据
        window: 时间窗口定义
        logger: 日志记录器

    返回:
        (X_train, y_train, X_valid, y_valid) 四个 DataFrame
    """
    # 对齐因子和标签的 index（两者行数可能不同）
    common_index = factors.index.intersection(labels.index)
    factors = factors.loc[common_index]
    labels = labels.loc[common_index]

    # 提取日期和股票代码
    dates = factors.index.get_level_values("date")
    codes = factors.index.get_level_values("code")

    # ----- 训练集 -----
    # 时间范围：train_start ~ train_end
    # 注意：需要剔除 embargo 期间的样本（它们的标签可能延伸到验证集）
    train_mask = (dates >= window.train_start) & (dates <= window.train_end)

    # 进一步剔除 embargo 期间的样本
    # 这些样本的标签（未来N日收益）可能跨越到验证集时间范围
    embargo_start = window.train_end + pd.Timedelta(days=window.embargo_days)
    train_mask = train_mask & (dates < window.train_end - pd.Timedelta(days=window.embargo_days))

    # ----- 验证集 -----
    valid_mask = (dates >= window.valid_start) & (dates <= window.valid_end)

    # 合并因子和标签
    # 只保留因子和标签都有值的样本
    common_cols = factors.columns.intersection(labels.columns)

    X_train = factors[train_mask]
    y_train = labels.loc[train_mask, "label"]

    X_valid = factors[valid_mask]
    y_valid = labels.loc[valid_mask, "label"]

    # 剔除 NaN
    train_valid = X_train.notna().all(axis=1) & y_train.notna()
    X_train = X_train[train_valid]
    y_train = y_train[train_valid]

    valid_valid = X_valid.notna().all(axis=1) & y_valid.notna()
    X_valid = X_valid[valid_valid]
    y_valid = y_valid[valid_valid]

    return X_train, y_train, X_valid, y_valid


def run_split(config: dict, logger) -> List[dict]:
    """
    执行完整的时间序列划分，返回所有窗口的训练/验证数据。

    参数:
        config: 配置字典
        logger: 日志记录器

    返回:
        列表，每个元素是一个 dict:
        {
            "window_id": int,
            "window": TimeWindow,
            "X_train": DataFrame,
            "y_train": Series,
            "X_valid": DataFrame,
            "y_valid": Series,
        }
    """
    proc_path = get_data_path(config, "processed")

    # 加载数据
    logger.info("加载预处理后的因子和标签数据...")
    factors = load_dataframe(proc_path / "factors_processed.parquet", logger)
    labels = load_dataframe(proc_path / "labels.parquet", logger)

    # 获取所有交易日
    dates = factors.index.get_level_values("date").unique().sort_values()
    logger.info(f"数据覆盖 {len(dates)} 个交易日: {dates[0].date()} ~ {dates[-1].date()}")

    # 生成滚动窗口
    windows = generate_rolling_windows(dates, config, logger)

    # 对每个窗口切分数据
    results = []
    for window in windows:
        X_train, y_train, X_valid, y_valid = split_data_by_window(
            factors, labels, window, logger
        )

        logger.info(f"  窗口 {window.window_id}: "
                     f"训练 {len(X_train)} 样本, 验证 {len(X_valid)} 样本")

        results.append({
            "window_id": window.window_id,
            "window": window,
            "X_train": X_train,
            "y_train": y_train,
            "X_valid": X_valid,
            "y_valid": y_valid,
        })

    logger.info(f"共生成 {len(results)} 个训练/验证窗口")
    return results


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="时间序列划分")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路径")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logger("split", config.get("log_level", "INFO"))

    logger.info("=" * 60)
    logger.info("时间序列滚动划分")
    logger.info("=" * 60)

    results = run_split(config, logger)

    logger.info("=" * 60)
    logger.info("时间序列划分完成！")
    for r in results:
        w = r["window"]
        logger.info(f"  窗口 {r['window_id']}: "
                     f"训练 {len(r['X_train'])} 样本 ({w.train_start.date()} ~ {w.train_end.date()}), "
                     f"验证 {len(r['X_valid'])} 样本 ({w.valid_start.date()} ~ {w.valid_end.date()})")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
