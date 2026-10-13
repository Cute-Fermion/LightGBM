"""
evaluation.py - 评估指标模块
===============================
计算量化选股模型的各项评估指标。

核心指标:
    - Rank IC：预测值与未来收益的横截面 Spearman 相关系数
    - ICIR：IC 均值 / IC 标准差（衡量 IC 的稳定性）
    - 年化收益率、年化波动率
    - 夏普比率（Sharpe Ratio）
    - 最大回撤（Max Drawdown）
    - Calmar 比率（年化收益 / 最大回撤）
    - 换手率
    - 分组单调性检验

运行方式:
    python src/evaluation.py --config config/config.yaml
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from scipy.stats import spearmanr

from utils import (
    load_config, setup_logger, PROJECT_ROOT,
    get_output_path, get_data_path, ensure_dir, save_dataframe, load_dataframe
)


# ============================================================
# 第一部分：IC 相关指标
# ============================================================

def calc_ic_series(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    logger=None
) -> pd.Series:
    """
    计算每日的 Rank IC 时间序列。

    Rank IC = Spearman 相关系数(预测值排名, 真实收益排名)

    参数:
        predictions: 预测值，MultiIndex (date, code)，列 'pred'
        labels: 标签数据，MultiIndex (date, code)，列 'label'
        logger: 日志记录器

    返回:
        IC 时间序列，index 为日期
    """
    # 合并预测值和标签
    merged = predictions[["pred"]].join(labels[["label"]], how="inner")

    # 逐日计算 IC
    ic_series = merged.groupby(level="date").apply(
        lambda g: spearmanr(g["pred"], g["label"])[0] if len(g) > 10 else np.nan
    )

    if logger:
        logger.info(f"  IC 序列: 均值={ic_series.mean():.4f}, "
                     f"标准差={ic_series.std():.4f}, "
                     f"非空率={ic_series.notna().mean():.2%}")

    return ic_series


def calc_icir(ic_series: pd.Series) -> float:
    """
    计算 ICIR（IC Information Ratio）。

    ICIR = IC 均值 / IC 标准差

    含义：IC 的稳定性。ICIR > 1 说明 IC 比较稳定，选股能力可靠。
    ICIR < 0.5 说明 IC 波动太大，模型不稳定。

    参数:
        ic_series: IC 时间序列

    返回:
        ICIR 值
    """
    if ic_series.std() == 0:
        return 0.0
    return ic_series.mean() / ic_series.std()


def calc_ic_decay(
    predictions: pd.DataFrame,
    labels_dict: Dict[int, pd.DataFrame],
    logger=None
) -> pd.Series:
    """
    计算 IC 衰减：不同持有期的 IC 变化。

    目的：观察模型的预测能力随时间衰减的速度。
    如果 1 日 IC 很高但 20 日 IC 接近 0，说明模型只做超短期预测。

    参数:
        predictions: 预测值
        labels_dict: {持有天数: 对应标签} 的字典
        logger: 日志记录器

    返回:
        IC 衰减序列，index 为持有天数
    """
    decay_results = {}

    for period, labels in labels_dict.items():
        ic = calc_ic_series(predictions, labels)
        decay_results[period] = ic.mean()

    result = pd.Series(decay_results)

    if logger:
        logger.info(f"  IC 衰减: {result.to_dict()}")

    return result


# ============================================================
# 第二部分：收益风险指标
# ============================================================

def calc_annual_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    计算年化收益率。

    公式: (1 + 总收益) ^ (252 / 实际天数) - 1

    参数:
        returns: 收益率序列
        periods_per_year: 每年的周期数（日频=252，月频=12）

    返回:
        年化收益率
    """
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    if n_periods == 0:
        return 0.0
    annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
    return annual_return


def calc_annual_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    计算年化波动率。

    公式: 日波动率 × √252

    参数:
        returns: 收益率序列
        periods_per_year: 每年的周期数

    返回:
        年化波动率
    """
    return returns.std() * np.sqrt(periods_per_year)


def calc_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.025,
    periods_per_year: int = 252
) -> float:
    """
    计算夏普比率（Sharpe Ratio）。

    夏普 = (年化收益 - 无风险利率) / 年化波动率

    含义：每承担一单位风险获得的超额收益。
    夏普 > 1 是不错的水平，> 2 非常优秀。

    参数:
        returns: 收益率序列
        risk_free_rate: 年化无风险利率（默认 2.5%）
        periods_per_year: 每年的周期数

    返回:
        夏普比率
    """
    annual_ret = calc_annual_return(returns, periods_per_year)
    annual_vol = calc_annual_volatility(returns, periods_per_year)
    if annual_vol == 0:
        return 0.0
    return (annual_ret - risk_free_rate) / annual_vol


def calc_max_drawdown(returns: pd.Series) -> float:
    """
    计算最大回撤（Maximum Drawdown）。

    最大回撤 = 从最高点到最低点的最大跌幅

    含义：最坏情况下的亏损幅度。
    最大回撤 < 20% 是比较好的水平。

    参数:
        returns: 收益率序列

    返回:
        最大回撤（负数，如 -0.15 表示 15% 回撤）
    """
    # 计算累计净值
    nav = (1 + returns).cumprod()
    # 计算历史最高点
    peak = nav.cummax()
    # 计算回撤
    drawdown = (nav - peak) / peak
    return drawdown.min()


def calc_calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    计算 Calmar 比率。

    Calmar = 年化收益 / |最大回撤|

    含义：每承担一单位回撤风险获得的年化收益。
    Calmar > 1 说明年化收益大于最大回撤，是不错的水平。

    参数:
        returns: 收益率序列
        periods_per_year: 每年的周期数

    返回:
        Calmar 比率
    """
    annual_ret = calc_annual_return(returns, periods_per_year)
    max_dd = calc_max_drawdown(returns)
    if max_dd == 0:
        return 0.0
    return annual_ret / abs(max_dd)


# ============================================================
# 第三部分：分组单调性检验
# ============================================================

def calc_group_monotonicity(
    grouped_returns: pd.DataFrame,
    n_groups: int = 5,
    logger=None
) -> Dict:
    """
    检验分组收益的单调性。

    好的选股模型应该满足：
    - Group 5（Top）收益 > Group 4 > ... > Group 1（Bottom）
    - 多空收益差（Group 5 - Group 1）显著为正

    参数:
        grouped_returns: 各组的收益率 DataFrame
        n_groups: 分组数量
        logger: 日志记录器

    返回:
        字典，包含各组平均收益和单调性检验结果
    """
    group_cols = [f"group_{i}" for i in range(1, n_groups + 1)]
    available_cols = [c for c in group_cols if c in grouped_returns.columns]

    if not available_cols:
        return {"monotonic": False, "error": "缺少分组收益列"}

    # 各组平均收益
    group_means = grouped_returns[available_cols].mean()

    # 检查单调性：是否严格递增
    values = group_means.values
    is_monotonic = all(values[i] < values[i + 1] for i in range(len(values) - 1))

    # 多空收益差
    long_short = values[-1] - values[0]

    result = {
        "group_means": group_means.to_dict(),
        "monotonic": is_monotonic,
        "long_short_diff": long_short,
    }

    if logger:
        logger.info(f"  分组平均收益: {group_means.to_dict()}")
        logger.info(f"  单调性: {'是' if is_monotonic else '否'}")
        logger.info(f"  多空收益差: {long_short:.4f}")

    return result


# ============================================================
# 第四部分：综合评估
# ============================================================

def evaluate_model(
    model_name: str,
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    backtest_result: pd.DataFrame,
    config: dict,
    logger
) -> Dict:
    """
    对单个模型进行全面评估。

    参数:
        model_name: 模型名称
        predictions: 预测结果
        labels: 标签数据
        backtest_result: 回测结果
        config: 配置字典
        logger: 日志记录器

    返回:
        评估结果字典
    """
    logger.info(f"\n评估模型: {model_name}")

    risk_free = config["evaluation"]["risk_free_rate"]
    n_groups = config["backtest"]["n_groups"]

    results = {"model_name": model_name}

    # ----- IC 指标 -----
    logger.info("  计算 IC 指标...")
    ic_series = calc_ic_series(predictions, labels, logger)
    results["ic_mean"] = ic_series.mean()
    results["ic_std"] = ic_series.std()
    results["icir"] = calc_icir(ic_series)
    results["ic_positive_ratio"] = (ic_series > 0).mean()  # IC > 0 的天数占比

    # ----- 收益风险指标（Top 组） -----
    if "long_only_top" in backtest_result.columns:
        logger.info("  计算 Top 组收益指标...")
        top_returns = backtest_result["long_only_top"]
        results["annual_return_top"] = calc_annual_return(top_returns)
        results["annual_vol_top"] = calc_annual_volatility(top_returns)
        results["sharpe_top"] = calc_sharpe_ratio(top_returns, risk_free)
        results["max_drawdown_top"] = calc_max_drawdown(top_returns)
        results["calmar_top"] = calc_calmar_ratio(top_returns)

    # ----- 收益风险指标（多空） -----
    if "long_short" in backtest_result.columns:
        logger.info("  计算多空组合收益指标...")
        ls_returns = backtest_result["long_short"]
        results["annual_return_ls"] = calc_annual_return(ls_returns)
        results["annual_vol_ls"] = calc_annual_volatility(ls_returns)
        results["sharpe_ls"] = calc_sharpe_ratio(ls_returns, risk_free)
        results["max_drawdown_ls"] = calc_max_drawdown(ls_returns)
        results["calmar_ls"] = calc_calmar_ratio(ls_returns)

    # ----- 分组单调性 -----
    logger.info("  检验分组单调性...")
    mono_result = calc_group_monotonicity(backtest_result, n_groups, logger)
    results["monotonic"] = mono_result["monotonic"]
    results["long_short_diff"] = mono_result.get("long_short_diff", 0)

    return results


def run_evaluation(config: dict, logger) -> pd.DataFrame:
    """
    对所有模型进行综合评估。

    参数:
        config: 配置字典
        logger: 日志记录器

    返回:
        评估结果 DataFrame（每行一个模型）
    """
    proc_path = get_data_path(config, "processed")
    pred_path = get_output_path(config, "results") / "predictions"
    back_path =  get_output_path(config, "results") / "backtest"
    output_path = get_output_path(config, "results") / "evaluation"
    ensure_dir(output_path)

    # 加载标签数据
    labels = load_dataframe(proc_path / "labels.parquet", logger)

    # 评估每个模型
    all_results = []

    pred_files = list(pred_path.glob("*_predictions.parquet"))
    for pred_file in pred_files:
        model_name = pred_file.stem.replace("_predictions", "")

        # 加载预测结果
        predictions = load_dataframe(pred_file, logger)

        # 加载回测结果
        backtest_file = back_path / f"backtest_{model_name}.parquet"
        if backtest_file.exists():
            backtest_result = load_dataframe(backtest_file, logger)
        else:
            logger.warning(f"  {model_name} 回测结果不存在，跳过")
            continue

        # 评估
        result = evaluate_model(model_name, predictions, labels, backtest_result, config, logger)
        all_results.append(result)

    if all_results:
        # 转为 DataFrame 并保存
        summary = pd.DataFrame(all_results)
        summary = summary.set_index("model_name")

        # 打印汇总表格
        logger.info("\n" + "=" * 80)
        logger.info("模型评估汇总")
        logger.info("=" * 80)
        for col in summary.columns:
            if summary[col].dtype in [np.float64, np.int64, float, int]:
                logger.info(f"  {col}:")
                for model, value in summary[col].items():
                    logger.info(f"    {model}: {value:.4f}")

        # 保存评估结果
        save_path = output_path / "evaluation_summary.parquet"
        save_dataframe(summary, save_path, logger)

        # 同时保存为 CSV（方便查看）
        summary.to_csv(output_path / "evaluation_summary.csv")
        logger.info(f"评估结果已保存: {save_path}")

        return summary
    else:
        logger.error("没有可评估的模型！")
        return pd.DataFrame()


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="模型评估")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路径")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logger("evaluation", config.get("log_level", "INFO"))

    logger.info("=" * 60)
    logger.info("模型评估开始")
    logger.info("=" * 60)

    summary = run_evaluation(config, logger)

    logger.info("\n" + "=" * 60)
    logger.info("评估完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
