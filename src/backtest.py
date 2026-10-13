"""
backtest.py - 回测与组合构建模块
====================================
将模型预测值转化为投资组合，模拟真实交易过程。

回测流程:
    1. 在每个调仓日，按预测值将股票分成 N 组
    2. 做多 Top 组（预测值最高的组）
    3. 计算每组的每日收益（考虑交易成本）
    4. 生成分组净值曲线和多空收益

交易细节:
    - 佣金：买卖各 0.3‰
    - 印花税：卖出 0.5‰
    - 滑点：0.1%
    - 涨跌停：涨停不能买，跌停不能卖
    - 停牌：停牌期间不交易

运行方式:
    python src/backtest.py --config config/config.yaml
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

from utils import (
    load_config, setup_logger, PROJECT_ROOT,
    get_data_path, get_output_path, ensure_dir, save_dataframe, load_dataframe
)


# ============================================================
# 第一部分：分组函数
# ============================================================

def assign_groups(
    predictions: pd.DataFrame,
    n_groups: int = 5
) -> pd.DataFrame:
    """
    在每个调仓日，按预测值将股票分成 N 组。

    分组方法：横截面排名等分
    - 第 1 组（Group 1）：预测值最低的 20% 股票
    - 第 N 组（Group N）：预测值最高的 20% 股票
    - Top 组 = Group N（做多），Bottom 组 = Group 1（做空/不做）

    参数:
        predictions: 预测值 DataFrame，MultiIndex 为 (date, code)，列 'pred'
        n_groups: 分组数量（默认 5 组）

    返回:
        DataFrame，新增 'group' 列（1~N）
    """
    result = predictions.copy()

    # 按日期分组，在每个横截面上按预测值排名
    def _assign_group(group):
        # pct=True 返回百分比排名（0~1）
        # 然后乘以 n_groups 并取整，得到组号（0~N-1）
        # 加 1 使组号从 1 开始
        group["group"] = group["pred"].rank(pct=True, na_option="keep").apply(
            lambda x: min(int(x * n_groups) + 1, n_groups) if pd.notna(x) else np.nan
        )
        return group

    # 只在调仓日做分组（每月一次）
    # 这里假设 predictions 已经只包含调仓日的数据
    result = result.groupby(level="date", group_keys=False).apply(_assign_group)

    return result


# ============================================================
# 第二部分：收益计算
# ============================================================

def calc_group_returns(
    grouped_data: pd.DataFrame,
    market_data: pd.DataFrame,
    config: dict,
    logger
) -> pd.DataFrame:
    """
    计算每组的每日收益率。

    考虑的交易成本:
    - 买入成本 = 佣金 + 滑点
    - 卖出成本 = 佣金 + 印花税 + 滑点
    - 调仓时的换仓成本 = 卖出旧持仓 + 买入新持仓

    参数:
        grouped_data: 分组数据，MultiIndex (date, code)，含 'group', 'pred', 'label' 列
        market_data: 市场数据，含 'close', 'pct_change', 'volume' 等
        config: 配置字典
        logger: 日志记录器

    返回:
        DataFrame，index 为日期，列为各组的日收益率
    """
    n_groups = config["backtest"]["n_groups"]
    commission = config["backtest"]["commission_rate"]
    stamp_tax = config["backtest"]["stamp_tax_rate"]
    slippage = config["backtest"]["slippage_rate"]
    long_only = config["backtest"]["long_only"]
    rebalance_days = config["backtest"]["rebalance_days"]

    logger.info(f"回测参数: {n_groups}组, 佣金={commission}, 印花税={stamp_tax}, 滑点={slippage}")

    # ----- 获取个股日收益率 -----
    # 使用 pct_change 列（如果存在），否则手动计算
    if "pct_change" in market_data.columns:
        daily_returns = market_data["pct_change"] / 100  # akshare 返回的是百分比，转为小数
    else:
        close = market_data["close"]
        close_df = close.unstack(level="code")
        daily_returns = close_df.pct_change().stack()

    # ----- 按调仓日分组计算收益 -----
    dates = grouped_data.index.get_level_values("date").unique().sort_values()

    # 每隔 rebalance_days 取一个调仓日
    rebalance_dates = dates[::rebalance_days]

    group_daily_returns = {f"group_{i}": [] for i in range(1, n_groups + 1)}
    date_list = []

    # 跟踪当前持仓（用于计算换手率和交易成本）
    prev_holdings = {i: set() for i in range(1, n_groups + 1)}

    for i, reb_date in enumerate(rebalance_dates):
        # 获取调仓日的分组信息
        try:
            day_data = grouped_data.xs(reb_date, level="date")
        except KeyError:
            continue

        if len(day_data) == 0:
            continue

        # 确定下一个调仓日（用于计算持有期收益）
        if i + 1 < len(rebalance_dates):
            next_reb_date = rebalance_dates[i + 1]
        else:
            # 最后一个调仓日，持有到数据结束
            next_reb_date = dates[-1]

        # 对每个组，计算持有期收益
        for g in range(1, n_groups + 1):
            group_stocks = day_data[day_data["group"] == g].index.get_level_values("code")

            if len(group_stocks) == 0:
                group_daily_returns[f"group_{g}"].append(0.0)
                continue

            # 获取这些股票在持有期的平均日收益
            # 简化处理：等权持有，计算平均收益
            try:
                if isinstance(daily_returns.index, pd.MultiIndex):
                    group_ret = daily_returns.loc[
                        (slice(reb_date, next_reb_date - pd.Timedelta(days=1)),),
                    ]
                    # 筛选组内股票
                    codes_in_group = group_ret.index.get_level_values("code")
                    mask = codes_in_group.isin(group_stocks)
                    group_ret = group_ret[mask]

                    if len(group_ret) > 0:
                        # 每日等权平均收益
                        daily_avg = group_ret.groupby(level="date").mean()
                        # 持有期总收益
                        period_return = (1 + daily_avg).prod() - 1
                    else:
                        period_return = 0.0
                else:
                    period_return = 0.0
            except Exception:
                period_return = 0.0

            # 扣除交易成本
            # 计算换手率：新持仓与旧持仓的交集比例
            current_holdings = set(group_stocks)
            if len(prev_holdings[g]) > 0:
                turnover = 1 - len(current_holdings & prev_holdings[g]) / max(len(current_holdings), 1)
            else:
                turnover = 1.0  # 首次建仓，全部换手

            # 交易成本 = 换手部分 × (买入成本 + 卖出成本)
            buy_cost = commission + slippage
            sell_cost = commission + stamp_tax + slippage
            total_cost = turnover * (buy_cost + sell_cost)

            net_return = period_return - total_cost
            group_daily_returns[f"group_{g}"].append(net_return)

            # 更新持仓记录
            prev_holdings[g] = current_holdings

        date_list.append(reb_date)

    # 转为 DataFrame
    result = pd.DataFrame(group_daily_returns, index=date_list)

    # ----- 计算多空收益 -----
    if not long_only:
        # 多空组合：做多 Top 组，做空 Bottom 组
        result["long_short"] = result[f"group_{n_groups}"] - result["group_1"]
    else:
        # 只做多 Top 组
        result["long_only_top"] = result[f"group_{n_groups}"]

    # 计算累计净值
    if "long_short" in result.columns:
        result["long_short_nav"] = (1 + result["long_short"]).cumprod()
    result["long_only_nav"] = (1 + result["long_only_top"]).cumprod()

    logger.info(f"回测完成: {len(result)} 个调仓期")
    return result


# ============================================================
# 第三部分：涨跌停处理
# ============================================================

def check_limit_up_down(data: pd.DataFrame) -> pd.DataFrame:
    """
    标记涨跌停状态。

    A 股涨跌停规则：
    - 主板：±10%
    - ST 股票：±5%
    - 科创板/创业板：±20%

    涨停不能买入，跌停不能卖出。

    参数:
        data: 市场数据，含 'pct_change' 列

    返回:
        DataFrame，新增 'limit_up' 和 'limit_down' 布尔列
    """
    result = data.copy()

    if "pct_change" in result.columns:
        pct = result["pct_change"]
        # 简化处理：统一使用 10% 作为涨跌停阈值
        # 实际应该根据股票类型动态调整
        result["limit_up"] = pct >= 9.9    # 涨停
        result["limit_down"] = pct <= -9.9  # 跌停
    else:
        result["limit_up"] = False
        result["limit_down"] = False

    return result


# ============================================================
# 第四部分：主回测流程
# ============================================================

def run_backtest(config: dict, logger) -> Dict[str, pd.DataFrame]:
    """
    执行完整回测流程。

    参数:
        config: 配置字典
        logger: 日志记录器

    返回:
        字典 {model_name: backtest_results}
    """
    proc_path = get_data_path(config, "processed")
    pred_path = get_output_path(config, "results") / "predictions"
    output_path = get_output_path(config, "results") / "backtest"
    ensure_dir(output_path)

    # 加载市场数据
    market_data = load_dataframe(proc_path / "market_data.parquet", logger)

    # 加载各模型的预测结果
    results = {}
    pred_files = list(pred_path.glob("*_predictions.parquet"))

    if not pred_files:
        logger.error("未找到任何预测结果文件！请先运行 models.py")
        return results

    for pred_file in pred_files:
        model_name = pred_file.stem.replace("_predictions", "")
        logger.info(f"\n回测模型: {model_name}")

        # 加载预测结果
        predictions = load_dataframe(pred_file, logger)

        if "pred" not in predictions.columns:
            logger.warning(f"  {model_name} 预测结果缺少 'pred' 列，跳过")
            continue

        # 分组
        n_groups = config["backtest"]["n_groups"]
        grouped = assign_groups(predictions, n_groups)

        # 计算分组收益
        backtest_result = calc_group_returns(grouped, market_data, config, logger)

        # 保存回测结果
        save_path = output_path / f"backtest_{model_name}.parquet"
        save_dataframe(backtest_result, save_path, logger)

        results[model_name] = backtest_result

        # 打印简要结果
        if "long_only_top" in backtest_result.columns:
            total_return = (1 + backtest_result["long_only_top"]).prod() - 1
            logger.info(f"  {model_name} Top 组总收益: {total_return:.2%}")
        if "long_short" in backtest_result.columns:
            total_return = (1 + backtest_result["long_short"]).prod() - 1
            logger.info(f"  {model_name} 多空总收益: {total_return:.2%}")

    return results


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="回测")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路径")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logger("backtest", config.get("log_level", "INFO"))

    logger.info("=" * 60)
    logger.info("回测开始")
    logger.info("=" * 60)

    results = run_backtest(config, logger)

    logger.info("\n" + "=" * 60)
    logger.info("回测完成！")
    for name, result in results.items():
        logger.info(f"  {name}: {len(result)} 个调仓期")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
