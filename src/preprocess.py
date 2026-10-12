"""
preprocess.py - 因子预处理模块
===============================
对原始因子进行三步处理：去极值 → 标准化 → 中性化。

为什么要做预处理？
    原始因子的分布千差万别：
    - 有的因子有极端值（如 PE 可能为负几百）
    - 有的因子量纲不同（如市值是百亿级别，收益率是个位数百分比）
    - 有的因子跟市值/行业高度相关（不中性化会导致模型只学到"大盘股"特征）

    预处理后，所有因子都变成：
    - 无极端值（去极值）
    - 均值=0，标准差=1（标准化）
    - 与市值和行业无关（中性化）

处理流程:
    1. 去极值：MAD 法，将超过 中位数 ± 3×MAD 的值截断
    2. 标准化：横截面 z-score 标准化
    3. 中性化：对市值（对数）和行业哑变量做回归，取残差

运行方式:
    python src/preprocess.py --config config/config.yaml
"""

import argparse
import numpy as np
import pandas as pd
from typing import Optional

from utils import (
    load_config, setup_logger, PROJECT_ROOT,
    get_data_path, save_dataframe, load_dataframe
)


# ============================================================
# 第一步：去极值
# ============================================================

def winsorize_mad(series: pd.Series, n: float = 3.0) -> pd.Series:
    """
    MAD 法去极值（Median Absolute Deviation）。

    为什么用 MAD 而不是均值±标准差？
    因为 MAD 基于中位数，对极端值不敏感（robust）。
    如果用均值±标准差，极端值本身就会拉偏均值和标准差，导致去极值失效。

    算法:
        median = 中位数
        MAD = 中位数(|x - median|)
        上界 = median + n × 1.4826 × MAD
        下界 = median - n × 1.4826 × MAD
        （1.4826 是 MAD 到标准差的转换系数，假设正态分布）

    参数:
        series: 单个横截面的因子值序列
        n: MAD 倍数阈值（默认 3 倍）

    返回:
        去极值后的序列（极端值被截断到边界）
    """
    median = series.median()
    mad = (series - median).abs().median()
    # 1.4826 是将 MAD 转换为标准差估计的系数
    mad_e = 1.4826 * mad
    upper = median + n * mad_e
    lower = median - n * mad_e
    return series.clip(lower, upper)


def winsorize_cross_section(factors: pd.DataFrame, config: dict, logger) -> pd.DataFrame:
    """
    对每个交易日（横截面）的每个因子做去极值。

    参数:
        factors: 因子 DataFrame，MultiIndex 为 (date, code)
        config: 配置字典
        logger: 日志记录器

    返回:
        去极值后的因子 DataFrame
    """
    n = config["factors"]["processing"]["winsorize_n"]  # 默认 3.0
    logger.info(f"去极值：MAD 法，阈值 {n} 倍...")

    # 按日期分组，对每个因子做 MAD 去极值
    def _apply_winsorize(group):
        for col in group.columns:
            group[col] = winsorize_mad(group[col], n)
        return group

    result = factors.groupby(level="date", group_keys=False).apply(_apply_winsorize)
    return result


# ============================================================
# 第二步：标准化
# ============================================================

def standardize_zscore(series: pd.Series) -> pd.Series:
    """
    z-score 标准化：(x - mean) / std。

    标准化后：均值=0，标准差=1。
    这样不同因子之间可以比较大小。

    参数:
        series: 单个横截面的因子值序列

    返回:
        标准化后的序列
    """
    mean = series.mean()
    std = series.std()
    if std == 0 or np.isnan(std):
        return series * 0  # 标准差为 0 说明所有值相同，全部变为 0
    return (series - mean) / std


def standardize_rank(series: pd.Series) -> pd.Series:
    """
    Rank 标准化：将值替换为排名，再映射到 [0, 1]。

    优点：对极端值完全不敏感，分布变成均匀分布。
    缺点：丢失了原始数值的距离信息。

    参数:
        series: 单个横截面的因子值序列

    返回:
        排名标准化后的序列（0~1 之间）
    """
    ranked = series.rank(pct=True, na_option="keep")
    return ranked


def normalize_cross_section(factors: pd.DataFrame, config: dict, logger) -> pd.DataFrame:
    """
    对每个交易日（横截面）的每个因子做标准化。

    参数:
        factors: 因子 DataFrame，MultiIndex 为 (date, code)
        config: 配置字典
        logger: 日志记录器

    返回:
        标准化后的因子 DataFrame
    """
    method = config["factors"]["processing"]["normalize_method"]  # "zscore" 或 "rank"
    logger.info(f"标准化：{method} 方法...")

    if method == "zscore":
        normalize_func = standardize_zscore
    elif method == "rank":
        normalize_func = standardize_rank
    else:
        raise ValueError(f"不支持的标准化方法: {method}")

    def _apply_normalize(group):
        for col in group.columns:
            group[col] = normalize_func(group[col])
        return group

    result = factors.groupby(level="date", group_keys=False).apply(_apply_normalize)
    return result


# ============================================================
# 第三步：中性化
# ============================================================

def neutralize_cross_section(
    factors: pd.DataFrame,
    market_data: pd.DataFrame,
    industry_data: Optional[pd.DataFrame],
    config: dict,
    logger
) -> pd.DataFrame:
    """
    对每个交易日（横截面）的每个因子做中性化。

    中性化的目的：
    假设某个因子（如动量）跟市值高度相关，那模型可能只是学到了"大市值"的特征。
    中性化后，因子只剩下"与市值和行业无关"的纯 alpha 信号。

    方法：
    对每个因子做横截面回归：
        factor = β0 + β1 × ln_market_cap + β2 × industry_1 + ... + βk × industry_k + ε
    取残差 ε 作为中性化后的因子值。

    参数:
        factors: 因子 DataFrame，MultiIndex 为 (date, code)
        market_data: 市场数据（需要 total_mv 列用于市值）
        industry_data: 行业分类数据（code, industry）
        config: 配置字典
        logger: 日志记录器

    返回:
        中性化后的因子 DataFrame
    """
    logger.info("中性化：对市值和行业做线性回归，取残差...")

    # ----- 准备市值数据 -----
    # 使用对数市值作为回归变量（市值分布右偏，对数后接近正态）
    if "total_mv" in market_data.columns:
        ln_cap = np.log(market_data["total_mv"].replace(0, np.nan))
        ln_cap = ln_cap.rename("ln_market_cap")
    elif "ln_market_cap" in factors.columns:
        # 如果因子里已经有对数市值，直接用
        ln_cap = factors["ln_market_cap"]
    else:
        logger.warning("  无市值数据，跳过市值中性化")
        ln_cap = None

    # ----- 准备行业哑变量 -----
    industry_dummies = None
    if industry_data is not None and len(industry_data) > 0:
        # 将行业分类 merge 到因子数据上
        idx = factors.index
        codes = idx.get_level_values("code")

        # 创建 code -> industry 的映射
        industry_map = industry_data.set_index("code")["industry"]
        stock_industry = codes.map(industry_map)

        if stock_industry.notna().sum() > 0:
            # 创建行业哑变量（one-hot encoding）
            industry_dummies = pd.get_dummies(stock_industry, prefix="ind")
            industry_dummies.index = idx
            # 去掉第一个行业（避免完全多重共线性，即 dummy variable trap）
            if industry_dummies.shape[1] > 1:
                industry_dummies = industry_dummies.iloc[:, 1:]

    # ----- 逐日做回归取残差 -----
    # 构建回归变量矩阵
    regressors_list = []
    if ln_cap is not None:
        regressors_list.append(ln_cap.reindex(factors.index).rename("ln_market_cap"))
    if industry_dummies is not None:
        regressors_list.append(industry_dummies)

    if not regressors_list:
        logger.warning("  无回归变量，跳过中性化")
        return factors

    regressors = pd.concat(regressors_list, axis=1)

    # 逐日处理
    result_frames = []
    dates = factors.index.get_level_values("date").unique()

    for date in dates:
        # 取出当天的因子值和回归变量
        if isinstance(factors.index, pd.MultiIndex):
            factor_cross = factors.xs(date, level="date")
            reg_cross = regressors.xs(date, level="date") if date in regressors.index.get_level_values("date") else None
        else:
            continue

        if reg_cross is None or len(reg_cross) == 0:
            result_frames.append(factor_cross)
            continue

        # 对齐 index
        common_idx = factor_cross.index.intersection(reg_cross.index)
        if len(common_idx) < 10:
            # 样本太少，跳过中性化
            result_frames.append(factor_cross)
            continue

        factor_cross = factor_cross.loc[common_idx]
        reg_cross = reg_cross.loc[common_idx]

        # 转为 numpy 做回归（比 statsmodels 快）
        X = reg_cross.values.astype(float)
        # 添加截距项
        X = np.column_stack([np.ones(len(X)), X])

        neutralized = factor_cross.copy()
        for col in factor_cross.columns:
            y = factor_cross[col].values.astype(float)
            # 跳过全 NaN 的列
            valid_mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
            if valid_mask.sum() < 10:
                continue

            X_valid = X[valid_mask]
            y_valid = y[valid_mask]

            try:
                # OLS 回归：β = (X'X)^(-1) X'y
                # 残差：ε = y - Xβ
                beta, _, _, _ = np.linalg.lstsq(X_valid, y_valid, rcond=None)
                residuals = y_valid - X_valid @ beta

                # 将残差填回
                neutralized.loc[common_idx[valid_mask], col] = residuals
            except Exception:
                # 回归失败（如共线性），保留原值
                pass

        result_frames.append(neutralized)

    if result_frames:
        result = pd.concat(result_frames, keys=dates, names=["date", "code"])
        # 确保 index 与输入一致
        result = result.reindex(factors.index)
        return result
    else:
        return factors


# ============================================================
# 主函数：完整预处理流程
# ============================================================

def preprocess_factors(config: dict, logger) -> pd.DataFrame:
    """
    执行完整的因子预处理流程：去极值 → 标准化 → 中性化。

    参数:
        config: 配置字典
        logger: 日志记录器

    返回:
        预处理后的因子 DataFrame
    """
    proc_path = get_data_path(config, "processed")

    # 加载因子数据
    logger.info("加载原始因子数据...")
    factors = load_dataframe(proc_path / "factors.parquet", logger)
    logger.info(f"  原始因子形状: {factors.shape}")

    # 加载市场数据（用于中性化）
    market_data = load_dataframe(proc_path / "market_data.parquet", logger)

    # 加载行业数据（可选）
    industry_data = None
    if (proc_path / "industry.parquet").exists():
        industry_data = load_dataframe(proc_path / "industry.parquet", logger)

    # 步骤 1：去极值
    factors = winsorize_cross_section(factors, config, logger)

    # 步骤 2：标准化
    factors = normalize_cross_section(factors, config, logger)

    # 步骤 3：中性化
    factors = neutralize_cross_section(factors, market_data, industry_data, config, logger)

    # 保存预处理后的因子
    save_path = proc_path / "factors_processed.parquet"
    save_dataframe(factors, save_path, logger)

    logger.info(f"预处理完成，因子形状: {factors.shape}")
    return factors


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="因子预处理")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路径")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logger("preprocess", config.get("log_level", "INFO"))

    logger.info("=" * 60)
    logger.info("因子预处理开始")
    logger.info("=" * 60)

    factors = preprocess_factors(config, logger)

    logger.info("=" * 60)
    logger.info("因子预处理完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
