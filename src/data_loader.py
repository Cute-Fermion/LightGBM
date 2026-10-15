"""
data_loader.py - 数据获取与清洗模块
=====================================
负责从 akshare 获取所有原始数据，并保存为本地 parquet 文件。
后续所有流程直接读取本地文件，不再重复请求网络。

主要功能:
    1. 获取沪深300/中证500成分股列表
    2. 逐只下载个股日频行情（OHLCV + 换手率）
    3. 下载估值指标（PE/PB/PS/股息率/市值）
    4. 下载财务指标（ROE/ROA/毛利率/净利率/负债率/流动比率/营收增速）
    5. 下载指数日线行情
    6. 获取申万一级行业分类
    7. 数据清洗：剔除上市不足N日、停牌等

运行方式:
    python src/data_loader.py --config config/config.yaml
"""

import os
import sys
import time
import argparse
import traceback
from pathlib import Path
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
import pandas as pd
import akshare as ak
from tqdm import tqdm

# 导入项目工具函数
from utils import (
    load_config, setup_logger, PROJECT_ROOT,
    get_data_path, ensure_dir, save_dataframe, load_dataframe,
    init_run
)

# 线程锁，用于安全写入进度文件
_progress_lock = threading.Lock()


def _safe_write_progress(progress_path: Path, code: str):
    """
    线程安全地写入进度文件。
    """
    with _progress_lock:
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(f"{code}\n")


def _check_data_completeness(
    save_path: Path,
    expected_codes: List[str],
    logger,
    data_name: str = "数据"
) -> tuple:
    """
    检查本地数据文件是否完整覆盖了所有预期的股票代码。

    参数:
        save_path: 数据文件路径
        expected_codes: 预期的股票代码列表
        logger: 日志记录器
        data_name: 数据名称（用于日志）

    返回:
        (is_complete, existing_data, existing_codes)
        - is_complete: 是否完整
        - existing_data: 已有数据（如果不完整）
        - existing_codes: 已有数据的股票代码集合
    """
    if not save_path.exists():
        return False, None, set()

    try:
        existing_data = load_dataframe(save_path, logger)
        if existing_data is None or len(existing_data) == 0:
            return False, None, set()

        # 获取已有数据的股票代码
        if isinstance(existing_data.index, pd.MultiIndex) and "code" in existing_data.index.names:
            existing_codes = set(existing_data.index.get_level_values("code").unique())
        elif "code" in existing_data.columns:
            existing_codes = set(existing_data["code"].unique())
        else:
            logger.warning(f"{data_name} 数据中没有 code 列或 code 索引")
            return False, existing_data, set()

        expected_set = set(expected_codes)
        missing_codes = expected_set - existing_codes

        if len(missing_codes) == 0:
            logger.info(f"{data_name} 数据完整，覆盖 {len(existing_codes)} 只股票")
            return True, existing_data, existing_codes
        else:
            logger.warning(
                f"{data_name} 数据不完整: 已有 {len(existing_codes)} 只，"
                f"缺少 {len(missing_codes)} 只"
            )
            if len(missing_codes) <= 10:
                logger.warning(f"  缺少的股票: {missing_codes}")
            return False, existing_data, existing_codes

    except Exception as e:
        logger.warning(f"加载 {data_name} 数据失败: {e}，将重新下载")
        return False, None, set()


# ============================================================
# 第一部分：网络请求工具（重试 + 限速）
# ============================================================

def retry_request(func, max_retries: int = 3, sleep_seconds: float = 2.0, **kwargs):
    """
    带重试机制的函数调用包装器。

    akshare 的底层是爬虫，网络请求可能因为超时、限频等原因失败。
    本函数在失败时自动重试，每次重试间隔递增。

    参数:
        func: 要调用的函数（通常是 akshare 的接口函数）
        max_retries: 最大重试次数
        sleep_seconds: 首次重试的等待秒数，之后每次翻倍
        **kwargs: 传给 func 的参数

    返回:
        func 的返回值

    异常:
        如果所有重试都失败，抛出最后一次异常
    """
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return func(**kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                # 等待时间递增：2s -> 4s -> 8s
                wait_time = sleep_seconds * (2 ** attempt)
                time.sleep(wait_time)
            else:
                raise last_exception


# ============================================================
# 第二部分：成分股列表获取
# ============================================================

def get_index_constituents(config: dict, logger) -> pd.DataFrame:
    """
    获取指数成分股列表。

    使用 akshare 获取沪深300或中证500的当期成分股。
    注意：akshare 不提供完整的历史成分股数据，这里用当期成分股近似。
    这意味着 2018 年被踢出指数的股票不会被包含，存在一定的幸存者偏差。

    参数:
        config: 配置字典
        logger: 日志记录器

    返回:
        DataFrame，包含 'code'（股票代码）和 'name'（股票名称）列
    """
    save_path = get_data_path(config, "raw") / "constituents.parquet"

    # 如果本地已有数据，直接加载
    if save_path.exists():
        logger.info(f"从本地加载成分股列表: {save_path}")
        return load_dataframe(save_path, logger)

    universe = config["data"]["universe"]
    logger.info(f"正在获取 {universe} 成分股列表...")

    try:
        if universe == "hs300":
            # 沪深300成分股：akshare 接口
            df = retry_request(ak.index_stock_cons, symbol="000300")
            # akshare 返回的列名可能是 '品种代码' / '品种名称' 等
            # 统一重命名为英文列名
            df = df.rename(columns={
                "品种代码": "code",
                "品种名称": "name",
                "stock_code": "code",
                "stock_name": "name",
                "constituent_code": "code",
                "constituent_name": "name",
            })
        elif universe == "zz500":
            # 中证500成分股
            df = retry_request(ak.index_stock_cons, symbol="000905")
            df = df.rename(columns={
                "品种代码": "code",
                "品种名称": "name",
                "stock_code": "code",
                "stock_name": "name",
                "constituent_code": "code",
                "constituent_name": "name",
            })
        else:
            raise ValueError(f"不支持的股票池: {universe}，请使用 hs300 或 zz500")

        # 确保 code 列是 6 位字符串（如 "000001"）
        if "code" in df.columns:
            df["code"] = df["code"].astype(str).str.zfill(6)
        else:
            # 如果列名不匹配，尝试取第一列作为 code
            logger.warning(f"成分股数据列名未匹配: {df.columns.tolist()}，尝试使用第一列")
            df.columns = ["code", "name"] + df.columns[2:].tolist()
            df["code"] = df["code"].astype(str).str.zfill(6)

        logger.info(f"获取到 {len(df)} 只成分股")
        save_dataframe(df, save_path, logger)
        return df

    except Exception as e:
        logger.error(f"获取成分股列表失败: {e}")
        logger.error(traceback.format_exc())
        raise


# ============================================================
# 第三部分：个股日频行情下载
# ============================================================

def download_daily_prices(
    config: dict,
    stock_codes: List[str],
    logger
) -> pd.DataFrame:
    """
    逐只下载个股日频行情数据。

    数据来源：东方财富（通过 akshare 的 stock_zh_a_hist 接口）。
    包含字段：日期、开盘、收盘、最高、最低、成交量、成交额、换手率、涨跌幅。
    使用**前复权**价格，保证价格连续性。

    下载策略：
    - 支持断点续拉：如果中途中断，再次运行会跳过已下载的股票
    - 每只股票下载间隔 0.3 秒，避免被限频
    - 失败自动重试 3 次

    参数:
        config: 配置字典
        stock_codes: 股票代码列表
        logger: 日志记录器

    返回:
        合并后的 DataFrame，MultiIndex 为 (date, code)
    """
    save_path = get_data_path(config, "raw") / "daily_prices.parquet"
    progress_path = get_data_path(config, "raw") / "_price_download_progress.txt"

    # 确保目录存在
    ensure_dir(get_data_path(config, "raw"))

    # 检查本地数据是否完整
    is_complete, existing_data, existing_codes = _check_data_completeness(
        save_path, stock_codes, logger, "行情数据"
    )
    if is_complete:
        return existing_data

    # 确定需要下载的股票列表
    if existing_data is not None and len(existing_codes) > 0:
        # 数据不完整，只下载缺失的股票
        codes_to_download = [c for c in stock_codes if c not in existing_codes]
        logger.info(f"数据不完整，需要补下载 {len(codes_to_download)} 只股票")
        # 已有数据的股票视为已下载
        downloaded = set(existing_codes)
    else:
        codes_to_download = stock_codes
        downloaded = set()
        # 读取进度文件（断点续拉）
        if progress_path.exists():
            with open(progress_path, "r") as f:
                downloaded = set(line.strip() for line in f.readlines())
            codes_to_download = [c for c in stock_codes if c not in downloaded]
            logger.info(f"发现已有进度，已下载 {len(downloaded)} 只，剩余 {len(codes_to_download)} 只")

    start_date = config["data"]["start_date"].replace("-", "")
    end_date = config["data"]["end_date"].replace("-", "")

    new_data = []
    failed_codes = []

    logger.info(f"开始下载 {len(codes_to_download)} 只股票的日频行情...")

    for code in tqdm(codes_to_download, desc="下载行情数据"):
        try:
            # akshare 个股日频行情（东财源）
            # adjust="qfq" 表示前复权
            df = retry_request(
                ak.stock_zh_a_hist,
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )

            if df is not None and len(df) > 0:
                # 统一列名（akshare 返回中文列名）
                df = df.rename(columns={
                    "日期": "date",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "amount",
                    "换手率": "turnover_rate",
                    "涨跌幅": "pct_change",
                    "振幅": "amplitude",
                })

                # 添加股票代码列
                df["code"] = code

                # 转换日期格式
                df["date"] = pd.to_datetime(df["date"])

                new_data.append(df)

                # 记录进度（线程安全）
                _safe_write_progress(progress_path, code)

            # 限速：每次请求间隔 0.3 秒，避免被封 IP
            time.sleep(0.3)

        except Exception as e:
            logger.warning(f"下载 {code} 失败: {e}")
            failed_codes.append(code)
            # 失败后等久一点再试下一只
            time.sleep(1.0)
            continue

    # 处理下载结果
    downloaded_count = len(existing_codes) + len(new_data)

    if new_data:
        # 合并新下载的数据与已有数据
        new_df = pd.concat(new_data, ignore_index=True)

        if existing_data is not None and len(existing_data) > 0:
            # 合并已有数据和新数据
            existing_reset = existing_data.reset_index()
            result = pd.concat([existing_reset, new_df], ignore_index=True)
        else:
            result = new_df

        result = result.drop_duplicates(subset=["date", "code"])
        result = result.set_index(["date", "code"]).sort_index()

        logger.info(
            f"行情数据: {len(result)} 条记录，"
            f"覆盖 {result.index.get_level_values('code').nunique()} 只股票"
        )
        if failed_codes:
            logger.warning(f"下载失败的股票 ({len(failed_codes)} 只): {failed_codes[:10]}")

        # 只有全部下载成功才删除进度文件
        if len(failed_codes) == 0 and downloaded_count >= len(stock_codes):
            save_dataframe(result, save_path, logger)
            if progress_path.exists():
                progress_path.unlink()
            logger.info("所有股票下载完成，进度文件已清理")
        else:
            # 部分成功：保存已有+新下载的数据，保留进度文件以便下次续拉
            save_dataframe(result, save_path, logger)
            logger.warning(f"部分下载完成，已保存 {len(result)} 条记录，下次运行将补下载缺失股票")

        return result

    elif existing_data is not None and len(existing_data) > 0:
        # 没有新数据下载（全部失败或无需下载），返回已有数据
        logger.warning("本次未下载任何新数据，返回已有数据")
        return existing_data
    else:
        logger.error("没有成功下载任何数据！")
        return pd.DataFrame()


# ============================================================
# 第四部分：估值指标下载
# ============================================================

def download_valuation_indicators(
    config: dict,
    stock_codes: List[str],
    logger
) -> pd.DataFrame:
    """
    下载个股估值指标：PE、PB、PS、总市值。

    数据来源：百度股市通（通过 akshare 的 stock_zh_valuation_baidu 接口）。
    这些指标是日频的，对因子计算非常重要。

    注意：新版 akshare 已移除 stock_a_indicator_lg，改用 stock_zh_valuation_baidu。
    该接口需要分指标调用（总市值、市盈率TTM、市净率、市销率），每个指标一次请求。

    参数:
        config: 配置字典
        stock_codes: 股票代码列表（纯数字，如 '000001'）
        logger: 日志记录器

    返回:
        DataFrame，MultiIndex 为 (date, code)
    """
    save_path = get_data_path(config, "raw") / "valuation.parquet"
    progress_path = get_data_path(config, "raw") / "_valuation_download_progress.txt"

    # 确保目录存在
    ensure_dir(get_data_path(config, "raw"))

    # 检查本地数据是否完整
    is_complete, existing_data, existing_codes = _check_data_completeness(
        save_path, stock_codes, logger, "估值数据"
    )
    if is_complete:
        return existing_data

    # 确定需要下载的股票列表
    if existing_data is not None and len(existing_codes) > 0:
        codes_to_download = [c for c in stock_codes if c not in existing_codes]
        logger.info(f"估值数据不完整，需要补下载 {len(codes_to_download)} 只股票")
        downloaded = set(existing_codes)
    else:
        downloaded = set()
        # 读取进度文件
        if progress_path.exists():
            with open(progress_path, "r") as f:
                downloaded = set(line.strip() for line in f.readlines())
            logger.info(f"估值数据已有进度: {len(downloaded)} 只")
        codes_to_download = [c for c in stock_codes if c not in downloaded]

    all_data = []
    failed_codes = []

    # 需要下载的指标列表
    # 百度接口支持的指标：总市值、市盈率(TTM)、市盈率(静)、市净率、市销率
    indicators = {
        "总市值": "total_mv",      # 总市值（亿元）
        "市盈率(TTM)": "pe_ttm",   # 市盈率(TTM)
        "市净率": "pb",            # 市净率
        "市销率": "ps_ttm",        # 市销率
    }

    logger.info(f"开始下载 {len(codes_to_download)} 只股票的估值指标...")
    logger.info(f"使用 stock_zh_valuation_baidu，每只股票需要 {len(indicators)} 次请求")

    for code in tqdm(codes_to_download, desc="下载估值指标"):
        try:
            # 提取纯数字代码（akshare 百度接口需要纯数字）
            pure_code = code.split('.')[0] if '.' in code else code
            # 去掉可能的交易所后缀
            pure_code = pure_code.replace('SH', '').replace('SZ', '').replace('sh', '').replace('sz', '')

            stock_data = {"code": code, "date": None}
            dates_set = None

            # 逐个指标下载
            for indicator_name, col_name in indicators.items():
                try:
                    df = retry_request(
                        ak.stock_zh_valuation_baidu,
                        symbol=pure_code,
                        indicator=indicator_name,
                        period="近三年"  # 覆盖项目所需时间范围
                    )

                    if df is not None and len(df) > 0:
                        df["date"] = pd.to_datetime(df["date"])
                        df = df.rename(columns={"value": col_name})

                        # 记录日期列
                        if dates_set is None:
                            stock_data["date"] = df["date"]
                            dates_set = set(df["date"])
                        else:
                            # 只保留所有指标都有的日期
                            current_dates = set(df["date"])
                            dates_set = dates_set & current_dates

                        stock_data[col_name] = df.set_index("date")["value"]

                    time.sleep(0.2)  # 限速

                except Exception as e:
                    logger.debug(f"  {code} 指标 {indicator_name} 下载失败: {e}")
                    continue

            # 合并该股票的所有指标
            if dates_set and len(dates_set) > 0:
                # 直接从各指标的 Series 合并
                merged = pd.DataFrame({"date": sorted(dates_set), "code": code})
                merged["date"] = pd.to_datetime(merged["date"])

                for col_name in indicators.values():
                    if col_name in stock_data and isinstance(stock_data[col_name], pd.Series):
                        series = stock_data[col_name]
                        series.index = pd.to_datetime(series.index)
                        merged[col_name] = merged["date"].map(series)

                # 只保留配置的时间范围
                start = pd.to_datetime(config["data"]["start_date"])
                end = pd.to_datetime(config["data"]["end_date"])
                merged = merged[(merged["date"] >= start) & (merged["date"] <= end)]

                # 至少要有市值数据
                if "total_mv" in merged.columns and merged["total_mv"].notna().sum() > 0:
                    all_data.append(merged)

                    # 记录进度（线程安全）
                    _safe_write_progress(progress_path, code)

            time.sleep(0.3)

        except Exception as e:
            logger.warning(f"下载 {code} 估值指标失败: {e}")
            failed_codes.append(code)
            time.sleep(1.0)
            continue

    # 处理下载结果
    downloaded_count = len(existing_codes) + len(all_data)

    if all_data:
        new_df = pd.concat(all_data, ignore_index=True)

        if existing_data is not None and len(existing_data) > 0:
            existing_reset = existing_data.reset_index()
            result = pd.concat([existing_reset, new_df], ignore_index=True)
        else:
            result = new_df

        result = result.drop_duplicates(subset=["date", "code"])
        result = result.set_index(["date", "code"]).sort_index()

        logger.info(f"估值数据: {len(result)} 条记录")
        if failed_codes:
            logger.warning(f"估值下载失败 {len(failed_codes)} 只: {failed_codes[:10]}")

        if len(failed_codes) == 0 and downloaded_count >= len(stock_codes):
            save_dataframe(result, save_path, logger)
            if progress_path.exists():
                progress_path.unlink()
            logger.info("所有股票估值数据下载完成，进度文件已清理")
        else:
            save_dataframe(result, save_path, logger)
            logger.warning(f"部分估值数据下载完成，已保存，下次运行将补下载缺失股票")

        return result

    elif existing_data is not None and len(existing_data) > 0:
        logger.warning("本次未下载任何新估值数据，返回已有数据")
        return existing_data
    else:
        logger.error("估值数据下载失败！")
        return pd.DataFrame()


# ============================================================
# 第五部分：财务指标下载
# ============================================================

def download_financial_data(
    config: dict,
    stock_codes: List[str],
    logger
) -> pd.DataFrame:
    """
    下载个股财务指标。

    由于 akshare 的财务数据接口按季度发布，我们需要：
    1. 获取每只股票的财务指标时间序列
    2. 按发布日对齐（使用固定延迟假设近似）

    财务指标包括：ROE、ROA、毛利率、净利率、负债率、流动比率、营收增速。

    参数:
        config: 配置字典
        stock_codes: 股票代码列表
        logger: 日志记录器

    返回:
        DataFrame，包含财务指标，按 (date, code) 对齐
    """
    save_path = get_data_path(config, "raw") / "financial_data.parquet"
    progress_path = get_data_path(config, "raw") / "_financial_download_progress.txt"

    # 确保目录存在
    ensure_dir(get_data_path(config, "raw"))

    # 检查本地数据是否完整
    is_complete, existing_data, existing_codes = _check_data_completeness(
        save_path, stock_codes, logger, "财务数据"
    )
    if is_complete:
        return existing_data

    # 确定需要下载的股票列表
    if existing_data is not None and len(existing_codes) > 0:
        codes_to_download = [c for c in stock_codes if c not in existing_codes]
        logger.info(f"财务数据不完整，需要补下载 {len(codes_to_download)} 只股票")
        downloaded = set(existing_codes)
    else:
        downloaded = set()
        if progress_path.exists():
            with open(progress_path, "r") as f:
                downloaded = set(line.strip() for line in f.readlines())
            logger.info(f"财务数据已有进度: {len(downloaded)} 只")
        codes_to_download = [c for c in stock_codes if c not in downloaded]

    logger.info(f"开始下载 {len(codes_to_download)} 只股票的财务指标...")

    all_data = []
    failed_codes = []
    delay_config = config["data"]["financial_report_delay"]

    for code in tqdm(codes_to_download, desc="下载财务数据"):
        try:
            # 获取财务分析指标
            df = retry_request(
                ak.stock_financial_analysis_indicator,
                symbol=code
            )

            if df is not None and len(df) > 0:
                # 统一列名（akshare 返回中文列名，且列名可能随版本变化）
                col_map = {}
                for col in df.columns:
                    col_lower = col.lower()
                    if "净资产收益率" in col or "roe" in col_lower:
                        col_map[col] = "roe"
                    elif "总资产报酬率" in col or "roa" in col_lower:
                        col_map[col] = "roa"
                    elif "销售毛利率" in col or "毛利率" in col:
                        col_map[col] = "gross_margin"
                    elif "销售净利率" in col or "净利率" in col:
                        col_map[col] = "net_margin"
                    elif "资产负债率" in col or "负债率" in col:
                        col_map[col] = "debt_ratio"
                    elif "流动比率" in col:
                        col_map[col] = "current_ratio"
                    elif "营业收入增长率" in col or "营收增速" in col:
                        col_map[col] = "revenue_growth"

                df = df.rename(columns=col_map)
                df["code"] = code

                # 尝试识别日期列
                date_col = None
                for col in df.columns:
                    if "日期" in col or "date" in col.lower() or "报告期" in col:
                        date_col = col
                        break

                if date_col:
                    df["report_date"] = pd.to_datetime(df[date_col], errors="coerce")

                    # 根据报告月份推算发布日（固定延迟假设）
                    # 例如：年报(12月) -> 发布日 = 报告日 + 120天
                    df["publish_date"] = df["report_date"].apply(
                        lambda d: _estimate_publish_date(d, delay_config) if pd.notna(d) else pd.NaT
                    )

                    all_data.append(df)

                    # 记录进度（线程安全）
                    _safe_write_progress(progress_path, code)

            time.sleep(0.5)

        except Exception as e:
            logger.warning(f"下载 {code} 财务数据失败: {e}")
            failed_codes.append(code)
            continue

    # 处理下载结果
    downloaded_count = len(existing_codes) + len(all_data)

    if all_data:
        new_df = pd.concat(all_data, ignore_index=True)

        if existing_data is not None and len(existing_data) > 0:
            result = pd.concat([existing_data, new_df], ignore_index=True)
        else:
            result = new_df

        result = result.drop_duplicates()

        logger.info(f"财务数据: {len(result)} 条记录")
        if failed_codes:
            logger.warning(f"财务数据下载失败 {len(failed_codes)} 只: {failed_codes[:10]}")

        if len(failed_codes) == 0 and downloaded_count >= len(stock_codes):
            save_dataframe(result, save_path, logger)
            if progress_path.exists():
                progress_path.unlink()
            logger.info("所有股票财务数据下载完成，进度文件已清理")
        else:
            save_dataframe(result, save_path, logger)
            logger.warning(f"部分财务数据下载完成，已保存，下次运行将补下载缺失股票")

        return result

    elif existing_data is not None and len(existing_data) > 0:
        logger.warning("本次未下载任何新财务数据，返回已有数据")
        return existing_data
    else:
        logger.error("财务数据下载失败！")
        return pd.DataFrame()


def _estimate_publish_date(report_date: pd.Timestamp, delay_config: dict) -> pd.Timestamp:
    """
    根据报告日期推算实际发布日（固定延迟假设）。

    A 股财报发布有季节性：
    - 年报（12-31 截止）-> 大多在次年 3-4 月发布，假设延迟 120 天
    - 半年报（6-30 截止）-> 大多在 7-8 月发布，假设延迟 60 天
    - Q1（3-31 截止）-> 大多在 4 月发布，假设延迟 30 天
    - Q3（9-30 截止）-> 大多在 10 月发布，假设延迟 30 天

    参数:
        report_date: 报告期末日期
        delay_config: 各类报告的延迟天数配置

    返回:
        估算的发布日
    """
    month = report_date.month
    day = report_date.day

    # 判断是哪种报告
    if month == 12 and day >= 28:
        # 年报
        delay = delay_config["annual"]
    elif month == 9 and day >= 28:
        # Q3
        delay = delay_config["q3"]
    elif month == 6 and day >= 28:
        # 半年报
        delay = delay_config["semi_annual"]
    elif month == 3 and day >= 28:
        # Q1
        delay = delay_config["q1"]
    else:
        # 默认延迟 60 天
        delay = 60

    return report_date + pd.Timedelta(days=delay)


# ============================================================
# 第六部分：指数行情下载
# ============================================================

def download_index_data(config: dict, logger) -> pd.DataFrame:
    """
    下载基准指数的日线行情（用于计算超额收益）。

    参数:
        config: 配置字典
        logger: 日志记录器

    返回:
        DataFrame，index 为日期，包含 close 列
    """
    save_path = get_data_path(config, "raw") / "index_daily.parquet"

    if save_path.exists():
        logger.info(f"从本地加载指数数据: {save_path}")
        return load_dataframe(save_path, logger)

    universe = config["data"]["universe"]
    if universe == "hs300":
        symbol = "000300"
        index_name = "沪深300"
    else:
        symbol = "000905"
        index_name = "中证500"

    logger.info(f"正在下载 {index_name} 指数行情...")

    try:
        # 使用 akshare 的指数日线行情接口
        df = retry_request(
            ak.stock_zh_index_daily,
            symbol=f"sh{symbol}"
        )

        if df is not None and len(df) > 0:
            df = df.rename(columns={
                "date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            })
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

            # 只保留配置的时间范围（多取一些数据用于计算因子）
            start = pd.to_datetime(config["data"]["start_date"]) - pd.Timedelta(days=400)
            end = pd.to_datetime(config["data"]["end_date"])
            df = df[(df.index >= start) & (df.index <= end)]

            logger.info(f"指数数据下载完成: {len(df)} 个交易日")
            save_dataframe(df, save_path, logger)
            return df
        else:
            logger.error("指数数据为空！")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"下载指数数据失败: {e}")
        raise


# ============================================================
# 第七部分：行业分类获取
# ============================================================

def download_industry_classification(config: dict, stock_codes: List[str], logger) -> pd.DataFrame:
    """
    获取申万一级行业分类。

    用于因子中性化时的行业哑变量构建。

    参数:
        config: 配置字典
        stock_codes: 股票代码列表
        logger: 日志记录器

    返回:
        DataFrame，包含 code 和 industry 列
    """
    save_path = get_data_path(config, "raw") / "industry_classification.parquet"

    if save_path.exists():
        logger.info(f"从本地加载行业分类: {save_path}")
        return load_dataframe(save_path, logger)

    logger.info("正在获取申万一级行业分类...")

    try:
        # 获取申万一级行业列表
        industry_list = []

        # 方法1：尝试通过 akshare 获取个股行业分类
        # 使用 stock_board_industry_cons_em 获取每个行业的成分股
        board_df = retry_request(ak.stock_board_industry_name_em, board_type="industry")

        if board_df is not None and len(board_df) > 0:
            # 遍历每个行业板块，获取成分股
            for _, row in tqdm(board_df.iterrows(), total=len(board_df), desc="获取行业成分"):
                board_name = row.get("板块名称", row.iloc[1] if len(row) > 1 else None)
                if board_name is None:
                    continue

                try:
                    cons_df = retry_request(
                        ak.stock_board_industry_cons_em,
                        symbol=board_name
                    )
                    if cons_df is not None and len(cons_df) > 0:
                        # 提取股票代码列
                        code_col = None
                        for col in cons_df.columns:
                            if "代码" in col or "code" in col.lower():
                                code_col = col
                                break
                        if code_col is None:
                            code_col = cons_df.columns[1]  # 通常第二列是代码

                        codes = cons_df[code_col].astype(str).str.zfill(6).tolist()
                        for code in codes:
                            if code in stock_codes:
                                industry_list.append({"code": code, "industry": board_name})

                    time.sleep(0.2)
                except Exception:
                    continue

        if industry_list:
            result = pd.DataFrame(industry_list).drop_duplicates(subset=["code"])
            logger.info(f"获取到 {len(result)} 只股票的行业分类，覆盖 {result['industry'].nunique()} 个行业")
            save_dataframe(result, save_path, logger)
            return result
        else:
            logger.warning("行业分类获取失败，将使用空数据")
            return pd.DataFrame(columns=["code", "industry"])

    except Exception as e:
        logger.error(f"获取行业分类失败: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame(columns=["code", "industry"])


# ============================================================
# 第八部分：数据清洗
# ============================================================

def clean_data(config: dict, logger) -> Dict[str, pd.DataFrame]:
    """
    对所有原始数据进行清洗，输出处理后的数据。

    清洗规则：
    1. 剔除上市不足 N 日的股票
    2. 剔除停牌日（成交量为 0 或 NaN）
    3. 合并行情、估值、财务数据

    参数:
        config: 配置字典
        logger: 日志记录器

    返回:
        字典，包含清洗后的各类数据
    """
    raw_path = get_data_path(config, "raw")
    proc_path = get_data_path(config, "processed")
    min_list_days = config["data"]["min_list_days"]

    # 检查必要的数据文件是否存在
    required_files = ["daily_prices.parquet", "valuation.parquet", "index_daily.parquet"]
    for f in required_files:
        if not (raw_path / f).exists():
            logger.error(f"必要的数据文件不存在: {raw_path / f}")
            logger.error("请先运行数据下载: python src/data_loader.py --config config/config.yaml")
            raise FileNotFoundError(f"缺少数据文件: {f}")

    # 加载数据
    logger.info("加载原始数据...")
    prices = load_dataframe(raw_path / "daily_prices.parquet", logger)
    valuation = load_dataframe(raw_path / "valuation.parquet", logger)
    index_data = load_dataframe(raw_path / "index_daily.parquet", logger)

    # 加载可选数据
    financial_data = None
    if (raw_path / "financial_data.parquet").exists():
        financial_data = load_dataframe(raw_path / "financial_data.parquet", logger)

    industry = None
    if (raw_path / "industry_classification.parquet").exists():
        industry = load_dataframe(raw_path / "industry_classification.parquet", logger)

    # ----- 清洗步骤 1：剔除上市不足 N 日的股票 -----
    logger.info(f"剔除上市不足 {min_list_days} 日的股票...")
    # 统计每只股票的交易天数
    stock_trade_days = prices.groupby(level="code").size()
    valid_stocks = stock_trade_days[stock_trade_days >= min_list_days].index
    prices = prices.loc[prices.index.get_level_values("code").isin(valid_stocks)]
    valuation = valuation.loc[valuation.index.get_level_values("code").isin(valid_stocks)]
    logger.info(f"保留 {len(valid_stocks)} 只股票（剔除 {stock_trade_days.count() - len(valid_stocks)} 只）")

    # ----- 清洗步骤 2：剔除停牌日 -----
    logger.info("剔除停牌日（成交量为 0 的记录）...")
    if "volume" in prices.columns:
        suspended = prices["volume"] == 0
        n_suspended = suspended.sum()
        prices = prices[~suspended]
        logger.info(f"剔除 {n_suspended} 条停牌记录")

    # ----- 清洗步骤 3：合并估值数据到行情数据 -----
    logger.info("合并行情与估值数据...")
    # prices 和 valuation 都是 (date, code) MultiIndex
    # 直接 join
    merged = prices.join(valuation, how="left")

    # ----- 保存清洗后的数据 -----
    logger.info("保存清洗后的数据...")
    save_dataframe(merged, proc_path / "market_data.parquet", logger)
    save_dataframe(index_data, proc_path / "index_data.parquet", logger)

    if financial_data is not None:
        save_dataframe(financial_data, proc_path / "financial_data.parquet", logger)
    if industry is not None:
        save_dataframe(industry, proc_path / "industry.parquet", logger)

    result = {
        "market_data": merged,
        "index_data": index_data,
    }
    if financial_data is not None:
        result["financial_data"] = financial_data
    if industry is not None:
        result["industry"] = industry

    return result


# ============================================================
# 第九部分：主函数（命令行入口）
# ============================================================

def main():
    """
    命令行入口：执行完整的数据获取与清洗流程。

    用法:
        python src/data_loader.py --config config/config.yaml
        python src/data_loader.py --config config/config.yaml --skip-download  # 只做清洗
    """
    parser = argparse.ArgumentParser(description="数据获取与清洗")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--skip-download", action="store_true", help="跳过下载，只做数据清洗")
    parser.add_argument("--run_id", type=int, default=None, help="运行序号（默认自动递增）")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    run_id = init_run(config, args.run_id)
    logger = setup_logger("data_loader", config.get("log_level", "INFO"))

    logger.info("=" * 60)
    logger.info("多因子选股项目 - 数据获取与清洗")
    logger.info("=" * 60)

    if not args.skip_download:
        # 步骤 1：获取成分股列表
        constituents = get_index_constituents(config, logger)
        stock_codes = constituents["code"].tolist()

        # 步骤 2：下载日频行情
        daily_prices = download_daily_prices(config, stock_codes, logger)

        # 步骤 3：下载估值指标
        valuation = download_valuation_indicators(config, stock_codes, logger)

        # 步骤 4：下载财务数据
        financial_data = download_financial_data(config, stock_codes, logger)

        # 步骤 5：下载指数行情
        index_data = download_index_data(config, logger)

        # 步骤 6：获取行业分类
        industry = download_industry_classification(config, stock_codes, logger)

    # 步骤 7：数据清洗
    logger.info("开始数据清洗...")
    cleaned = clean_data(config, logger)

    logger.info("=" * 60)
    logger.info("数据获取与清洗全部完成！")
    logger.info(f"处理后数据保存在: {get_data_path(config, 'processed')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
