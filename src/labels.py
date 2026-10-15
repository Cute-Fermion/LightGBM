"""
labels.py - 标签构建模块
=========================
构建模型训练的目标标签：未来 N 日收益率（横截面标准化）。

标签构建逻辑:
    1. 计算每只股票未来 20 日（约一个月）的前瞻收益率
    2. 在每个横截面（每个交易日）上做标准化：
       label = (forward_return - 截面均值) / 截面标准差
    3. 标准化后的标签含义：该股票在未来 N 日相对于全市场的超额表现
       - label > 0: 跑赢市场平均
       - label < 0: 跑输市场平均

关键约束:
    - 标签计算使用了"未来数据"，这是不可避免的（因为它是预测目标）
    - 但标签只在训练时使用，且通过 purge/embargo 防止数据泄露
    - 标签的最后 N 天数据无法计算（因为没有"未来"数据），自动为 NaN

运行方式:
    python src/labels.py --config config/config.yaml
"""

import argparse
import numpy as np
import pandas as pd

from utils import (
    load_config, setup_logger, PROJECT_ROOT,
    get_data_path, save_dataframe, load_dataframe,
    init_run
)


def build_labels(config: dict, logger) -> pd.DataFrame:
    """
    构建横截面标准化的前瞻收益率标签。

    计算步骤:
    1. 将收盘价 pivot 成 日期×股票 矩阵
    2. 计算 N 日前瞻收益率: forward_return[t] = close[t+N] / close[t] - 1
    3. 对每个交易日（横截面）做标准化:
       label[t] = (forward_return[t] - mean[t]) / std[t]

    参数:
        config: 配置字典
        logger: 日志记录器

    返回:
        DataFrame，MultiIndex 为 (date, code)，单列 'label'
    """
    proc_path = get_data_path(config, "processed")
    forward_days = config["labels"]["forward_days"]  # 20

    logger.info(f"构建标签：未来 {forward_days} 日收益率（横截面标准化）")

    # 加载市场数据
    market_data = load_dataframe(proc_path / "market_data.parquet", logger)
    close = market_data["close"]

    # ----- 步骤 1：计算前瞻收益率 -----
    # pivot 成 日期×股票 矩阵，方便做 shift
    close_df = close.unstack(level="code")

    # forward_return[t] = close[t+N] / close[t] - 1
    # 注意：这里用了"未来"数据，但这是标签，不是特征，所以是合理的
    forward_ret = close_df.shift(-forward_days) / close_df - 1

    # ----- 步骤 2：横截面标准化 -----
    # 对每个交易日（每行），减去均值再除以标准差
    # 这样标签的分布是：均值=0，标准差=1
    # 含义：该股票跑赢/跑输市场多少个标准差
    row_mean = forward_ret.mean(axis=1)   # 每日横截面均值
    row_std = forward_ret.std(axis=1)     # 每日横截面标准差

    # 标准化：减去均值，除以标准差
    # 加 eps 防止除以 0（极端情况下所有股票收益相同）
    eps = 1e-8
    label_df = (forward_ret.sub(row_mean, axis=0)).div(row_std + eps, axis=0)

    # ----- 步骤 3：转回 MultiIndex 格式 -----
    labels = label_df.stack().rename("label")
    labels = labels.to_frame()

    # 统计信息
    total = len(labels)
    valid = labels["label"].notna().sum()
    logger.info(f"  标签总数: {total}，有效: {valid}，缺失: {total - valid}")
    logger.info(f"  标签均值: {labels['label'].mean():.4f}，标准差: {labels['label'].std():.4f}")

    # 同时保存原始前瞻收益率（不做标准化），回测时可能用到
    raw_forward = forward_ret.stack().rename("forward_return")
    labels["forward_return"] = raw_forward

    # 保存
    save_path = proc_path / "labels.parquet"
    save_dataframe(labels, save_path, logger)

    return labels


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="标签构建")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--run_id", type=int, default=None, help="运行序号（默认自动递增）")
    args = parser.parse_args()

    config = load_config(args.config)
    run_id = init_run(config, args.run_id)
    logger = setup_logger("labels", config.get("log_level", "INFO"))

    logger.info("=" * 60)
    logger.info("标签构建开始")
    logger.info("=" * 60)

    labels = build_labels(config, logger)

    logger.info("=" * 60)
    logger.info("标签构建完成！")
    logger.info(f"标签数据形状: {labels.shape}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
