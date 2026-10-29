"""
features.py - 因子计算模块
===========================
基于清洗后的行情数据，计算选股因子。

因子类别:
    1. 动量类：中期趋势跟踪（5/10/20/60日收益率、52周高点距离）
    2. 反转类：短期反转效应（1日收益率，A股短期反转显著）
    3. 波动类：风险度量（下行波动率）
    4. 量价类：资金流向（换手率、量比、量价相关性）
    5. 估值类：价值因子（PE/PB/PS倒数、股息率、市值对数）
    6. 质量类：基本面质量（ROE/ROA/毛利率/净利率/负债率/流动比率/营收增速
               + 每股经营现金流/ROE同比变化/SUE盈利超预期）
    7. 另类-北向资金：沪深港通持股占比
    8. 另类-分析师预期：卖方一致评级

核心原则:
    所有因子计算**只使用 t 时刻及之前的数据**，绝不使用未来信息。

运行方式:
    python src/features.py --config config/config.yaml
"""

import argparse
import numpy as np
import pandas as pd
from typing import Dict

from utils import (
    load_config, setup_logger, PROJECT_ROOT,
    get_data_path, save_dataframe, load_dataframe,
    init_run
)


# ============================================================
# 工具函数：横截面操作辅助
# ============================================================

def _pct_change_by_group(close_series: pd.Series, window: int, index) -> pd.Series:
    """
    按 (date, code) MultiIndex 计算每只股票的 N 日收益率。

    为什么不用 pct_change + groupby?
    因为 MultiIndex 的 groupby(level="code") 后做 pct_change 效率较低。
    这里先把数据 pivot 成 日期×股票 的矩阵，直接对整个矩阵做 shift，
    然后再 stack 回来，速度快很多。

    参数:
        close_series: 收盘价序列（MultiIndex 为 date×code）
        window: 回望窗口天数
        index: 原始 MultiIndex，用于对齐输出

    返回:
        N 日收益率序列
    """
    # pivot: 行=日期, 列=股票代码, 值=收盘价
    close_df = close_series.unstack(level="code")
    # 计算 N 日收益率: (today / N_days_ago) - 1
    ret_df = close_df / close_df.shift(window) - 1
    # stack 回 MultiIndex 格式
    return ret_df.stack().reindex(index)


def _rolling_std_by_group(series: pd.Series, window: int, index) -> pd.Series:
    """
    按股票计算滚动标准差。

    参数:
        series: 输入序列（如日收益率）
        window: 滚动窗口
        index: 原始 MultiIndex

    返回:
        滚动标准差序列
    """
    df = series.unstack(level="code")
    result = df.rolling(window=window, min_periods=max(1, window // 2)).std()
    return result.stack().reindex(index)


def _rolling_mean_by_group(series: pd.Series, window: int, index) -> pd.Series:
    """
    按股票计算滚动均值。

    参数:
        series: 输入序列
        window: 滚动窗口
        index: 原始 MultiIndex

    返回:
        滚动均值序列
    """
    df = series.unstack(level="code")
    result = df.rolling(window=window, min_periods=max(1, window // 2)).mean()
    return result.stack().reindex(index)


def _rolling_corr_by_group(
    series_a: pd.Series, series_b: pd.Series, window: int, index
) -> pd.Series:
    """
    按股票计算两个序列的滚动相关系数。

    用于计算量价相关性（收益率与成交量的相关系数）。

    参数:
        series_a: 第一个序列（如日收益率）
        series_b: 第二个序列（如成交量）
        window: 滚动窗口
        index: 原始 MultiIndex

    返回:
        滚动相关系数序列
    """
    df_a = series_a.unstack(level="code")
    df_b = series_b.unstack(level="code")
    result = df_a.rolling(window=window, min_periods=max(1, window // 2)).corr(df_b)
    return result.stack().reindex(index)


# ============================================================
# 第一类：动量因子
# ============================================================

def calc_momentum_factors(data: pd.DataFrame, config: dict, logger) -> pd.DataFrame:
    """
    计算动量类因子。

    动量因子基于"趋势延续"假设：过去涨得好的股票未来一段时间内继续涨。
    但在 A 股市场，动量效应主要体现在中期（20-60日），短期反而是反转。

    因子列表:
    - mom_5d / mom_10d / mom_20d / mom_60d: 过去 N 日收益率
    - mom_20d_std: 过去 20 日收益率标准差（动量的波动）
    - high_52w_dist: 52 周高点距离 = 当前价 / 过去252日最高价 - 1

    参数:
        data: 市场数据，MultiIndex 为 (date, code)
        config: 配置字典
        logger: 日志记录器

    返回:
        DataFrame，包含所有动量因子，index 与输入相同
    """
    logger.info("计算动量因子...")
    idx = data.index
    close = data["close"]
    windows = config["factors"]["momentum_windows"]  # [5, 10, 20, 60]
    factors = pd.DataFrame(index=idx)

    # 1. N 日收益率（动量）
    for w in windows:
        col_name = f"mom_{w}d"
        factors[col_name] = _pct_change_by_group(close, w, idx)
        logger.debug(f"  {col_name}: 非空率 {factors[col_name].notna().mean():.2%}")

    # 2. 20 日收益率标准差（动量的稳定性）
    # 先计算日收益率，再算滚动标准差
    daily_ret = _pct_change_by_group(close, 1, idx)
    factors["mom_20d_std"] = _rolling_std_by_group(daily_ret, 20, idx)

    # 3. 52 周高点距离
    # 公式: 当前价 / 过去252日最高价 - 1
    # 值越接近 0 说明越接近高点，值为负说明离高点有距离
    high_52w_window = config["factors"]["high_52w_window"]  # 252
    high_df = data["high"].unstack(level="code")
    rolling_max = high_df.rolling(window=high_52w_window, min_periods=60).max()
    factors["high_52w_dist"] = (close / rolling_max.stack().reindex(idx)) - 1

    logger.info(f"  动量因子计算完成，共 {len(factors.columns)} 个因子")
    return factors


# ============================================================
# 第二类：反转因子
# ============================================================

def calc_reversal_factors(data: pd.DataFrame, config: dict, logger) -> pd.DataFrame:
    """
    计算反转类因子。

    A 股市场的短期反转效应非常显著：过去 1-5 日跌得多的股票，未来几天倾向于反弹。
    这与 A 股散户占比高、过度反应有关。

    因子列表:
    - rev_1d: 过去 1 日收益率（短期反转）
    - rev_5d: 过去 5 日收益率（短期反转）

    参数:
        data: 市场数据
        config: 配置字典
        logger: 日志记录器

    返回:
        DataFrame，包含反转因子
    """
    logger.info("计算反转因子...")
    idx = data.index
    close = data["close"]
    factors = pd.DataFrame(index=idx)

    # 1 日反转
    factors["rev_1d"] = _pct_change_by_group(close, 1, idx)

    # # 5 日反转（该因子与 mom_5d 重复，已删去）
    # factors["rev_5d"] = _pct_change_by_group(close, 5, idx)

    logger.info(f"  反转因子计算完成，共 {len(factors.columns)} 个因子")
    return factors


# ============================================================
# 第三类：波动因子
# ============================================================

def calc_volatility_factors(data: pd.DataFrame, config: dict, logger) -> pd.DataFrame:
    """
    计算波动类因子。

    低波动异象（Low Volatility Anomaly）：低波动股票长期收益反而更高。
    下行波动率比总波动率更能衡量"风险"，因为上涨的波动对投资者是好事。

    因子列表:
    - vol_20d: 过去 20 日已实现波动率（日收益率标准差 × √252 年化）
    - atr_14d: 14 日 ATR（平均真实波幅）/ 收盘价，归一化
    - down_vol_20d: 过去 20 日下行波动率（只计算负收益的标准差）

    参数:
        data: 市场数据
        config: 配置字典
        logger: 日志记录器

    返回:
        DataFrame，包含波动因子
    """
    logger.info("计算波动因子...")
    idx = data.index
    close = data["close"]
    high = data["high"]
    low = data["low"]
    vol_window = config["factors"]["volatility_window"]  # 20
    atr_window = config["factors"]["atr_window"]  # 14
    factors = pd.DataFrame(index=idx)

    # 日收益率
    daily_ret = _pct_change_by_group(close, 1, idx)

    # # 1. 20 日已实现波动率（年化）（该因子与 mom_20d_std 重复，已删去）
    # # 先算日标准差，再年化
    # daily_std = _rolling_std_by_group(daily_ret, vol_window, idx)
    # factors["vol_20d"] = daily_std * np.sqrt(252)

    # # 2. ATR（Average True Range）/ 收盘价，归一化（该因子与 mom_20d_std 高度相关，已删去）
    # # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    # prev_close = close.groupby(level="code").shift(1)
    # tr1 = high - low
    # tr2 = (high - prev_close).abs()
    # tr3 = (low - prev_close).abs()
    # true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    #
    # # 14 日 ATR
    # atr = _rolling_mean_by_group(true_range, atr_window, idx)
    # # 用收盘价归一化，使其具有横截面可比性
    # factors["atr_14d"] = atr / close

    # 3. 下行波动率：只考虑负收益日的标准差
    # 将正收益替换为 0，只保留亏损部分
    negative_ret = daily_ret.where(daily_ret < 0, 0)
    down_std = _rolling_std_by_group(negative_ret, vol_window, idx)
    factors["down_vol_20d"] = down_std * np.sqrt(252)

    logger.info(f"  波动因子计算完成，共 {len(factors.columns)} 个因子")
    return factors


# ============================================================
# 第四类：量价因子
# ============================================================

def calc_volume_price_factors(data: pd.DataFrame, config: dict, logger) -> pd.DataFrame:
    """
    计算量价类因子。

    量价关系反映资金的流入流出：
    - 放量上涨 -> 资金流入，看涨
    - 缩量上涨 -> 上涨乏力
    - 量价相关性高 -> 价格变动与成交量高度关联

    因子列表:
    - turnover_rate: 换手率（直接使用原始数据）
    - volume_ratio: 量比 = 5日均量 / 20日均量
    - vol_price_corr: 过去 20 日收益率与成交量的相关系数

    参数:
        data: 市场数据
        config: 配置字典
        logger: 日志记录器

    返回:
        DataFrame，包含量价因子
    """
    logger.info("计算量价因子...")
    idx = data.index
    close = data["close"]
    volume = data["volume"]
    turnover = data["turnover_rate"] if "turnover_rate" in data.columns else None

    short_w = config["factors"]["volume_change_short"]   # 5
    long_w = config["factors"]["volume_change_long"]     # 20
    corr_w = config["factors"]["volume_price_corr_window"]  # 20
    factors = pd.DataFrame(index=idx)

    # 1. 换手率（直接使用）
    if turnover is not None:
        factors["turnover_rate"] = turnover

    # 2. 量比 = 5日均量 / 20日均量
    # 量比 > 1 说明近期成交放量
    vol_short = _rolling_mean_by_group(volume, short_w, idx)
    vol_long = _rolling_mean_by_group(volume, long_w, idx)
    factors["volume_ratio"] = vol_short / vol_long

    # 3. 量价相关性：20日收益率与成交量的滚动相关系数
    daily_ret = _pct_change_by_group(close, 1, idx)
    factors["vol_price_corr"] = _rolling_corr_by_group(daily_ret, volume, corr_w, idx)

    logger.info(f"  量价因子计算完成，共 {len(factors.columns)} 个因子")
    return factors


# ============================================================
# 第五类：估值因子
# ============================================================

def calc_valuation_factors(data: pd.DataFrame, config: dict, logger) -> pd.DataFrame:
    """
    计算估值类因子。

    价值因子的核心逻辑：低估值股票长期跑赢高估值股票。
    使用倒数形式（如 1/PE）而非 PE 本身，原因：
    - PE 为负（亏损）时倒数仍有效（为负值），而 PE 本身无法直接比较
    - 倒数形式让"便宜"的股票值更大，方向统一

    因子列表:
    - ep: 1/PE（盈利收益率，越高越便宜）
    - bp: 1/PB（账面市值比）
    - sp: 1/PS（营收市值比）
    - dv_ttm: 股息率(TTM)
    - ln_market_cap: 对数总市值

    参数:
        data: 市场数据（需包含 pe_ttm, pb, ps_ttm, dv_ttm, total_mv 列）
        config: 配置字典
        logger: 日志记录器

    返回:
        DataFrame，包含估值因子
    """
    logger.info("计算估值因子...")
    idx = data.index
    factors = pd.DataFrame(index=idx)

    # 1. EP = 1/PE_TTM（市盈率倒数）
    # PE 为负（亏损）时，EP 也为负，横截面上自然排在后面
    if "pe_ttm" in data.columns:
        pe = data["pe_ttm"].replace(0, np.nan)  # 避免除以 0
        factors["ep"] = 1.0 / pe
    elif "pe" in data.columns:
        pe = data["pe"].replace(0, np.nan)
        factors["ep"] = 1.0 / pe

    # 2. BP = 1/PB（市净率倒数）
    if "pb" in data.columns:
        pb = data["pb"].replace(0, np.nan)
        factors["bp"] = 1.0 / pb

    # 3. SP = 1/PS_TTM（市销率倒数）
    if "ps_ttm" in data.columns:
        ps = data["ps_ttm"].replace(0, np.nan)
        factors["sp"] = 1.0 / ps
    elif "ps" in data.columns:
        ps = data["ps"].replace(0, np.nan)
        factors["sp"] = 1.0 / ps

    # 4. 股息率(TTM)
    if "dv_ttm" in data.columns:
        factors["dv_ttm"] = data["dv_ttm"]
    elif "dv_ratio" in data.columns:
        factors["dv_ttm"] = data["dv_ratio"]

    # 5. 对数总市值
    # 使用对数的原因：市值分布极度右偏，对数后接近正态分布
    if "total_mv" in data.columns:
        factors["ln_market_cap"] = np.log(data["total_mv"].replace(0, np.nan))

    logger.info(f"  估值因子计算完成，共 {len(factors.columns)} 个因子")
    return factors


# ============================================================
# 第六类：质量因子（来自财务数据）
# ============================================================

def calc_quality_factors(
    data: pd.DataFrame,
    financial_data: pd.DataFrame,
    config: dict,
    logger
) -> pd.DataFrame:
    """
    计算质量类因子。

    质量因子衡量公司的基本面质量：
    - 高 ROE/ROA -> 盈利能力强
    - 高毛利率/净利率 -> 竞争优势强
    - 低负债率 -> 财务安全
    - 高流动比率 -> 短期偿债能力强
    - 高营收增速 -> 成长性好

    注意：财务数据是季度频的，需要按发布日对齐到日频数据上。
    对齐方法：对于每个交易日，使用该日之前最新发布的财务数据。

    参数:
        data: 日频市场数据
        financial_data: 财务数据（包含 report_date, publish_date, 各财务指标）
        config: 配置字典
        logger: 日志记录器

    返回:
        DataFrame，包含质量因子，index 与 data 相同
    """
    logger.info("计算质量因子...")
    idx = data.index
    factors = pd.DataFrame(index=idx)

    if financial_data is None or len(financial_data) == 0:
        logger.warning("  无财务数据，质量因子全部为 NaN")
        for col in ["roe", "roa", "gross_margin", "net_margin", "debt_ratio", "current_ratio", "revenue_growth"]:
            factors[col] = np.nan
        return factors

    # ----- 将季度财务数据对齐到日频 -----
    # 思路：
    # 1. 财务数据有 publish_date（估算的发布日）
    # 2. 对于每个 (date, code)，找到 publish_date <= date 的最新一条记录
    # 3. 这就是该交易日"可获得的"最新财务数据

    # 提取需要的财务指标列
    quality_cols = [
        "roe", "roa", "gross_margin", "net_margin", "debt_ratio",
        "current_ratio", "revenue_growth",
        "ocfps", "roe_yoy", "dt_netprofit_yoy",
    ]
    available_cols = [c for c in quality_cols if c in financial_data.columns]

    if not available_cols:
        logger.warning(f"  财务数据中没有找到质量因子列，可用列: {financial_data.columns.tolist()}")
        for col in quality_cols:
            factors[col] = np.nan
        return factors

    # 确保 financial_data 有 publish_date 和 code 列
    if "publish_date" not in financial_data.columns or "code" not in financial_data.columns:
        logger.warning("  财务数据缺少 publish_date 或 code 列，无法对齐")
        for col in quality_cols:
            factors[col] = np.nan
        return factors

    # 只保留有用的列
    fin_subset = financial_data[["code", "publish_date"] + available_cols].copy()
    fin_subset = fin_subset.dropna(subset=["publish_date"])
    # 确保 publish_date 为 datetime 类型（tushare 的 ann_date 原始为字符串）
    fin_subset["publish_date"] = pd.to_datetime(fin_subset["publish_date"], errors="coerce")
    fin_subset = fin_subset.dropna(subset=["publish_date"])
    fin_subset = fin_subset.sort_values(["code", "publish_date"])

    # 获取日频数据的日期和股票代码
    dates = idx.get_level_values("date").unique().sort_values()
    codes = idx.get_level_values("code").unique()

    # 对每只股票，将财务数据按发布日展开到日频
    # 使用 pd.merge_asof 按时间最近匹配（高效实现）
    market_daily = pd.DataFrame(index=idx).reset_index()

    aligned_results = []
    for code in codes:
        fin_code = fin_subset[fin_subset["code"] == code].sort_values("publish_date")
        if len(fin_code) == 0:
            continue

        market_code = market_daily[market_daily["code"] == code][["date", "code"]].copy()
        if len(market_code) == 0:
            continue

        # merge_asof: 对每个 market date，找到 publish_date <= date 的最新财务记录
        merged = pd.merge_asof(
            market_code.sort_values("date"),
            fin_code.sort_values("publish_date"),
            left_on="date",
            right_on="publish_date",
            by="code",
            direction="backward"  # 只使用过去的数据
        )
        aligned_results.append(merged)

    if aligned_results:
        aligned = pd.concat(aligned_results, ignore_index=True)
        aligned = aligned.set_index(["date", "code"])

        # 将对齐后的财务数据填入因子 DataFrame
        for col in available_cols:
            if col in aligned.columns:
                factors[col] = aligned[col]

    # 填充未覆盖的列为 NaN
    for col in quality_cols:
        if col not in factors.columns:
            factors[col] = np.nan

    # ----- SUE（盈利超预期）-----
    # 使用 dt_netprofit_yoy（扣非净利润同比变化）作为盈利超前的代理变量
    # 在"季节性随机游走"模型下，去年同期盈利即为预期，YoY 变化就是 surprise
    # 横截面标准化使不同时期的 SUE 具有可比性
    if "dt_netprofit_yoy" in factors.columns:
        sue_raw = factors["dt_netprofit_yoy"].copy()
        date_level = factors.index.get_level_values("date")
        sue_group = sue_raw.groupby(date_level)
        sue_mean = sue_group.transform("mean")
        sue_std = sue_group.transform("std")
        factors["sue"] = np.where(sue_std > 0, (sue_raw - sue_mean) / sue_std, 0.0)
    else:
        factors["sue"] = np.nan

    logger.info(f"  质量因子计算完成，共 {len(factors.columns)} 个因子")
    return factors


# ============================================================
# 第七类：北向资金持股比例（另类数据）
# ============================================================

def calc_northbound_ratio(data: pd.DataFrame, config: dict, logger) -> pd.DataFrame:
    """
    计算北向资金（沪深港通）持股比例因子。

    北向资金被视为"聪明资金"，其持股变动包含增量信息。
    持股比例上升可能预示外资看好，下降则可能预示风险。

    数据来源:
        data/raw/northbound_data.parquet（由 data_loader_tushare 下载）

    参数:
        data: 日频市场数据，MultiIndex 为 (date, code)
        config: 配置字典
        logger: 日志记录器

    返回:
        DataFrame，包含 northbound_ratio 列，index 与 data 相同
    """
    logger.info("计算北向资金持股比例因子...")
    idx = data.index
    factors = pd.DataFrame(index=idx)
    factors["northbound_ratio"] = np.nan

    raw_path = get_data_path(config, "raw")
    nb_path = raw_path / "northbound_data.parquet"

    if not nb_path.exists():
        logger.warning("  北向资金数据不存在，跳过")
        return factors

    try:
        nb_data = load_dataframe(nb_path, logger)
        if nb_data is None or len(nb_data) == 0:
            logger.warning("  北向资金数据为空，跳过")
            return factors

        if "northbound_ratio" in nb_data.columns:
            # 精确匹配
            common_idx = idx.intersection(nb_data.index)
            factors.loc[common_idx, "northbound_ratio"] = nb_data.loc[common_idx, "northbound_ratio"]

            # 前向填充：北向持股比例变化缓慢，用最近一个已知值填充缺失日
            # 先 unstack 成 (date × code) 矩阵，按日期前向填充，再 stack 回去
            nb_unstacked = nb_data["northbound_ratio"].unstack("code")
            nb_unstacked = nb_unstacked.reindex(
                idx.get_level_values("date").unique()
            ).sort_index()
            nb_filled = nb_unstacked.ffill(limit=30)  # 最多向前填充 30 天
            nb_filled = nb_filled.stack()
            nb_filled.index.names = ["date", "code"]

            common_idx_filled = idx.intersection(nb_filled.index)
            # 只在精确匹配为 NaN 的位置用填充值补充
            mask = factors["northbound_ratio"].isna()
            fill_values = nb_filled.reindex(common_idx_filled)
            fill_idx = common_idx_filled[mask.reindex(common_idx_filled).fillna(False)]
            factors.loc[fill_idx, "northbound_ratio"] = fill_values.loc[fill_idx]

        coverage = factors["northbound_ratio"].notna().mean()
        logger.info(f"  北向持股比例因子完成，非空率 {coverage:.2%}")
    except Exception as e:
        logger.warning(f"  加载北向资金数据失败: {e}")

    return factors


# ============================================================
# 第八类：分析师一致评级（另类数据）
# ============================================================

def calc_consensus_rating(data: pd.DataFrame, config: dict, logger) -> pd.DataFrame:
    """
    计算分析师一致评级因子。

    卖方分析师的综合评级反映市场对该股票的专业预期。
    评级越高（接近 5 = 买入），市场预期越乐观。

    数据来源:
        data/raw/consensus_ratings.parquet（由 data_loader_tushare 下载）
        评级映射: 买入/强烈推荐=5, 增持/优于大市=4, 中性=3, 减持=2, 卖出=1

    参数:
        data: 日频市场数据，MultiIndex 为 (date, code)
        config: 配置字典
        logger: 日志记录器

    返回:
        DataFrame，包含 consensus_rating 列，index 与 data 相同
    """
    logger.info("计算分析师一致评级因子...")
    idx = data.index
    factors = pd.DataFrame(index=idx)
    factors["consensus_rating"] = np.nan

    raw_path = get_data_path(config, "raw")
    cr_path = raw_path / "consensus_ratings.parquet"

    if not cr_path.exists():
        logger.warning("  分析师评级数据不存在，跳过")
        return factors

    try:
        cr_data = load_dataframe(cr_path, logger)
        if cr_data is None or len(cr_data) == 0:
            logger.warning("  分析师评级数据为空，跳过")
            return factors

        if "consensus_rating" in cr_data.columns:
            common_idx = idx.intersection(cr_data.index)
            factors.loc[common_idx, "consensus_rating"] = cr_data.loc[common_idx, "consensus_rating"]

        coverage = factors["consensus_rating"].notna().mean()
        logger.info(f"  分析师一致评级因子完成，非空率 {coverage:.2%}")
    except Exception as e:
        logger.warning(f"  加载分析师评级数据失败: {e}")

    return factors


# ============================================================
# 主函数：计算所有因子
# ============================================================

def compute_all_factors(config: dict, logger) -> pd.DataFrame:
    """
    计算所有因子并合并为一个 DataFrame。

    流程:
    1. 加载清洗后的市场数据
    2. 分别计算六大类因子
    3. 合并所有因子
    4. 保存结果

    参数:
        config: 配置字典
        logger: 日志记录器

    返回:
        合并后的因子 DataFrame，MultiIndex 为 (date, code)
    """
    proc_path = get_data_path(config, "processed")

    # 加载数据
    logger.info("加载清洗后的市场数据...")
    market_data = load_dataframe(proc_path / "market_data.parquet", logger)

    # 加载财务数据（可选）
    financial_data = None
    if (proc_path / "financial_data.parquet").exists():
        financial_data = load_dataframe(proc_path / "financial_data.parquet", logger)

    # 分别计算因子
    momentum = calc_momentum_factors(market_data, config, logger)
    reversal = calc_reversal_factors(market_data, config, logger)
    volatility = calc_volatility_factors(market_data, config, logger)
    vol_price = calc_volume_price_factors(market_data, config, logger)
    valuation = calc_valuation_factors(market_data, config, logger)
    quality = calc_quality_factors(market_data, financial_data, config, logger)
    northbound = calc_northbound_ratio(market_data, config, logger)
    consensus = calc_consensus_rating(market_data, config, logger)

    # 合并所有因子
    logger.info("合并所有因子...")
    all_factors = pd.concat([
        momentum, reversal, volatility, vol_price, valuation, quality,
        northbound, consensus
    ], axis=1)

    # 删除缺失比例 > 50% 的因子
    max_missing = config["factors"]["processing"]["max_missing_ratio"]
    missing_ratio = all_factors.isna().mean()
    bad_cols = missing_ratio[missing_ratio > max_missing].index.tolist()
    if bad_cols:
        logger.warning(f"  剔除缺失率 > {max_missing:.0%} 的因子: {bad_cols}")
        all_factors = all_factors.drop(columns=bad_cols)

    logger.info(f"最终因子数量: {len(all_factors.columns)}")
    logger.info(f"因子列表: {all_factors.columns.tolist()}")

    # 保存
    save_path = proc_path / "factors.parquet"
    save_dataframe(all_factors, save_path, logger)

    return all_factors


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="因子计算")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--run_id", type=int, default=None, help="运行序号（默认自动递增）")
    args = parser.parse_args()

    config = load_config(args.config)
    run_id = init_run(config, args.run_id)
    logger = setup_logger("features", config.get("log_level", "INFO"))

    logger.info("=" * 60)
    logger.info("因子计算开始")
    logger.info("=" * 60)

    factors = compute_all_factors(config, logger)

    logger.info("=" * 60)
    logger.info("因子计算完成！")
    logger.info(f"因子数据形状: {factors.shape}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
