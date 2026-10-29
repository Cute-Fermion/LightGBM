"""
data_loader_tushare.py - 基于 Tushare 的数据获取与清洗模块
============================================================
优先使用 tushare pro 接口获取数据，对于 tushare 无法覆盖的数据自动回退到 akshare。

与原 data_loader.py 保持完全相同的函数签名和输出格式，
下游模块（features.py、labels.py 等）无需任何修改。

Tushare 接口说明:
    - 需要 tushare pro 账号和 token（https://tushare.pro）
    - 学生权限（2000 积分）对应频率限制：200 次/分钟
    - 在 config.yaml 中配置 tushare.token

与原版的关键差异:
    1. daily / daily_basic / index_daily 等接口按交易日批量获取（一次调用获取全市场），
       比逐只下载快数十倍
    2. 前复权价格通过 adj_factor 手动计算
    3. 估值指标（PE/PB/市值等）来自 daily_basic，无需逐只请求
    4. 行业分类回退到 akshare（tushare 的申万分类接口需要更高积分）

运行方式:
    python src/data_loader_tushare.py --config config/config.yaml
"""

import os
import sys
import time
import argparse
import traceback
import threading
from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import tushare as ts
from tqdm import tqdm

# 回退数据源
import akshare as ak

from utils import (
    load_config, setup_logger, PROJECT_ROOT,
    get_data_path, ensure_dir, save_dataframe, load_dataframe,
    init_run
)


from dotenv import load_dotenv

load_dotenv()          # 加载 env 文件

# ============================================================
# Tushare 初始化与速率控制
# ============================================================

def _init_tushare(config: dict, logger) -> Optional[object]:
    """
    初始化 tushare pro API。

    参数:
        config: 配置字典
        logger: 日志记录器

    返回:
        tushare pro_api 对象，如果 token 未配置则返回 None
    """
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        logger.warning("未配置 tushare.token，将全部使用 akshare 回退源")
        return None

    try:
        ts.set_token(token)
        pro = ts.pro_api()
        # 测试连接
        pro.trade_cal(exchange='SSE', start_date='20230101', end_date='20230105')
        logger.info("Tushare API 初始化成功")
        return pro
    except Exception as e:
        logger.warning(f"Tushare API 初始化失败: {e}，将回退到 akshare")
        return None


class _RateLimiter:
    """
    线程安全的滑动窗口速率限制器。

    确保在任意 60 秒窗口内的 API 调用次数不超过上限。
    """

    def __init__(self, calls_per_minute: int = 200):
        self.calls_per_minute = calls_per_minute
        self._timestamps: List[float] = []
        self._lock = threading.Lock()

    def wait(self):
        """阻塞直到可以发送下一次请求。"""
        with self._lock:
            now = time.time()
            # 清理 60 秒前的记录
            self._timestamps = [t for t in self._timestamps if now - t < 60]

            if len(self._timestamps) >= self.calls_per_minute:
                # 需要等待到最早的记录过期
                sleep_time = 60.0 - (now - self._timestamps[0]) + 0.05
                if sleep_time > 0:
                    time.sleep(sleep_time)
                now = time.time()
                self._timestamps = [t for t in self._timestamps if now - t < 60]

            self._timestamps.append(now)


# 全局速率限制器（2000 积分 → 200 次/分钟）
_rate_limiter = _RateLimiter(calls_per_minute=180)  # 略低于上限，留安全余量


# ============================================================
# 辅助函数
# ============================================================

def _ts_code_to_code(ts_code: str) -> str:
    """
    将 tushare 格式的股票代码转为项目格式。
    '000001.SZ' → '000001'
    """
    return ts_code.split('.')[0]


def _code_to_ts_code(code: str) -> str:
    """
    将项目格式的股票代码转为 tushare 格式。
    '000001' → '000001.SZ'
    """
    code = str(code).zfill(6)
    if code.startswith(('6', '9')):
        return f"{code}.SH"
    else:
        return f"{code}.SZ"


def _wait():
    """等待速率限制器。"""
    _rate_limiter.wait()


def _ts_request(func, max_retries: int = 3, sleep_seconds: float = 2.0, **kwargs):
    """
    带速率控制和重试的 tushare API 调用。

    参数:
        func: tushare API 方法
        max_retries: 最大重试次数
        sleep_seconds: 首次重试等待秒数，之后每次翻倍
        **kwargs: 传给 func 的参数

    返回:
        func 的返回值（DataFrame 或 None）

    异常:
        所有重试失败后抛出最后一次异常
    """
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            _wait()
            result = func(**kwargs)
            return result
        except Exception as e:
            last_exception = e
            err_msg = str(e).lower()
            # 频率限制相关错误，等待更久
            if 'freq' in err_msg or 'limit' in err_msg or '每分钟' in err_msg:
                time.sleep(30)
            elif attempt < max_retries:
                time.sleep(sleep_seconds * (2 ** attempt))
            else:
                raise last_exception


def _get_trading_dates(pro, config: dict, logger) -> List[str]:
    """
    获取配置时间范围内的所有交易日列表。

    参数:
        pro: tushare pro API
        config: 配置字典
        logger: 日志记录器

    返回:
        交易日字符串列表，格式 ['20180102', '20180103', ...]
    """
    start = config["data"]["start_date"].replace("-", "")
    end = config["data"]["end_date"].replace("-", "")

    try:
        df = _ts_request(
            pro.trade_cal,
            exchange='SSE',
            start_date=start,
            end_date=end,
            is_open='1'
        )
        if df is not None and len(df) > 0:
            dates = sorted(df['cal_date'].tolist())
            logger.info(f"获取到 {len(dates)} 个交易日")
            return dates
    except Exception as e:
        logger.warning(f"获取交易日历失败: {e}")

    # 回退：生成工作日（不含周末，含节假日误差）
    logger.info("使用工作日作为交易日近似")
    date_range = pd.bdate_range(start=start, end=end)
    return [d.strftime('%Y%m%d') for d in date_range]


_progress_lock = threading.Lock()

def _safe_write_progress(progress_path: Path, text: str):
    """线程安全地写入进度文件。"""
    with _progress_lock:
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(f"{text}\n")


def _check_data_completeness(
    save_path: Path,
    expected_codes: List[str],
    logger,
    data_name: str = "数据"
) -> Tuple[bool, Optional[pd.DataFrame], Set[str]]:
    """
    检查本地数据文件是否完整覆盖了所有预期的股票代码。

    返回:
        (is_complete, existing_data, existing_codes)
    """
    if not save_path.exists():
        return False, None, set()

    try:
        existing_data = load_dataframe(save_path, logger)
        if existing_data is None or len(existing_data) == 0:
            return False, None, set()

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
            return False, existing_data, existing_codes

    except Exception as e:
        logger.warning(f"加载 {data_name} 数据失败: {e}，将重新下载")
        return False, None, set()


def _estimate_publish_date(report_date: pd.Timestamp, delay_config: dict) -> pd.Timestamp:
    """
    根据报告日期推算实际发布日（固定延迟假设）。

    A 股财报发布有季节性：
    - 年报（12-31 截止）→ 延迟 120 天
    - 半年报（6-30 截止）→ 延迟 60 天
    - Q1（3-31 截止）→ 延迟 30 天
    - Q3（9-30 截止）→ 延迟 30 天
    """
    month = report_date.month
    day = report_date.day

    if month == 12 and day >= 28:
        delay = delay_config["annual"]
    elif month == 9 and day >= 28:
        delay = delay_config["q3"]
    elif month == 6 and day >= 28:
        delay = delay_config["semi_annual"]
    elif month == 3 and day >= 28:
        delay = delay_config["q1"]
    else:
        delay = 60

    return report_date + pd.Timedelta(days=delay)


# ============================================================
# 第一部分：成分股列表获取
# ============================================================

def get_index_constituents(config: dict, logger) -> pd.DataFrame:
    """
    获取指数成分股列表。

    优先使用 tushare 的 index_weight 接口获取历史成分股（比 akshare 的当期成分股更准确，
    可减少幸存者偏差）。如果 tushare 不可用，回退到 akshare。

    参数:
        config: 配置字典
        logger: 日志记录器

    返回:
        DataFrame，包含 'code'（6位股票代码）和 'name'（股票名称）列
    """
    save_path = get_data_path(config, "raw") / "constituents.parquet"

    if save_path.exists():
        logger.info(f"从本地加载成分股列表: {save_path}")
        return load_dataframe(save_path, logger)

    ensure_dir(get_data_path(config, "raw"))

    universe = config["data"]["universe"]
    index_code = "000300.SH" if universe == "hs300" else "000905.SH"
    index_name = "沪深300" if universe == "hs300" else "中证500"

    pro = _init_tushare(config, logger)

    if pro is not None:
        try:
            logger.info(f"[tushare] 获取 {index_name} 成分股列表 (index_weight)...")

            # 获取时间范围内最新的成分股
            # index_weight 需要 120 积分
            end_date = config["data"]["end_date"].replace("-", "")
            start_date = config["data"]["start_date"].replace("-", "")

            # 先尝试获取起始日期的成分股（更接近历史真实成分）
            df = None
            for try_date in [start_date, end_date]:
                try:
                    df = _ts_request(
                        pro.index_weight,
                        index_code=index_code,
                        start_date=try_date,
                        end_date=try_date
                    )
                    if df is not None and len(df) > 0:
                        break
                except Exception:
                    continue

            if df is not None and len(df) > 0:
                # 提取唯一的成分股代码
                codes = df['con_code'].unique()
                codes_list = [_ts_code_to_code(c) for c in codes]

                # 获取股票名称
                try:
                    stock_basic = _ts_request(
                        pro.stock_basic,
                        ts_code=','.join(codes[:50]),  # 分批查询
                        fields='ts_code,name'
                    )
                    if stock_basic is not None and len(stock_basic) > 0:
                        # 构建 code → name 映射
                        name_map = {}
                        for _, row in stock_basic.iterrows():
                            name_map[_ts_code_to_code(row['ts_code'])] = row['name']

                        # 如果第一批没覆盖全部，继续查
                        remaining = [c for c in codes_list if c not in name_map]
                        while remaining:
                            batch = remaining[:50]
                            remaining = remaining[50:]
                            try:
                                more = _ts_request(
                                    pro.stock_basic,
                                    ts_code=','.join([_code_to_ts_code(c) for c in batch]),
                                    fields='ts_code,name'
                                )
                                if more is not None:
                                    for _, row in more.iterrows():
                                        name_map[_ts_code_to_code(row['ts_code'])] = row['name']
                            except Exception:
                                pass

                        result = pd.DataFrame({
                            "code": codes_list,
                            "name": [name_map.get(c, "") for c in codes_list]
                        })
                    else:
                        result = pd.DataFrame({
                            "code": codes_list,
                            "name": [""] * len(codes_list)
                        })
                except Exception:
                    result = pd.DataFrame({
                        "code": codes_list,
                        "name": [""] * len(codes_list)
                    })

                result["code"] = result["code"].astype(str).str.zfill(6)
                logger.info(f"[tushare] 获取到 {len(result)} 只成分股")
                save_dataframe(result, save_path, logger)
                return result

        except Exception as e:
            logger.warning(f"[tushare] 获取成分股失败: {e}，回退到 akshare")

    # 回退到 akshare
    logger.info(f"[akshare] 获取 {index_name} 成分股列表...")
    try:
        symbol = "000300" if universe == "hs300" else "000905"
        df = ak.index_stock_cons(symbol=symbol)
        df = df.rename(columns={
            "品种代码": "code", "品种名称": "name",
            "stock_code": "code", "stock_name": "name",
            "constituent_code": "code", "constituent_name": "name",
        })

        if "code" not in df.columns:
            df.columns = ["code", "name"] + df.columns[2:].tolist()

        df["code"] = df["code"].astype(str).str.zfill(6)
        logger.info(f"[akshare] 获取到 {len(df)} 只成分股")
        save_dataframe(df, save_path, logger)
        return df

    except Exception as e:
        logger.error(f"获取成分股列表失败: {e}")
        raise


# ============================================================
# 第二部分：个股日频行情下载
# ============================================================

def download_daily_prices(
    config: dict,
    stock_codes: List[str],
    logger
) -> pd.DataFrame:
    """
    下载个股日频行情数据（前复权）。

    Tushare 策略：
        按交易日批量获取。每次调用 daily(trade_date=xxx) 获取当天全市场数据，
        再按 adj_factor 计算前复权价格。比逐只下载快约 300 倍。

    回退策略：
        tushare 不可用时，使用 akshare 的 stock_zh_a_hist 逐只下载。

    参数:
        config: 配置字典
        stock_codes: 股票代码列表
        logger: 日志记录器

    返回:
        合并后的 DataFrame，MultiIndex 为 (date, code)
    """
    save_path = get_data_path(config, "raw") / "daily_prices.parquet"
    progress_path = get_data_path(config, "raw") / "_price_download_progress.txt"

    ensure_dir(get_data_path(config, "raw"))

    # 检查完整性
    is_complete, existing_data, existing_codes = _check_data_completeness(
        save_path, stock_codes, logger, "行情数据"
    )
    if is_complete:
        return existing_data

    pro = _init_tushare(config, logger)

    if pro is not None:
        result = _download_daily_prices_tushare(
            pro, config, stock_codes, existing_data, existing_codes,
            progress_path, save_path, logger
        )
        if result is not None and len(result) > 0:
            return result
        logger.warning("[tushare] 行情数据下载失败，回退到 akshare")

    # 回退到 akshare
    return _download_daily_prices_akshare(
        config, stock_codes, existing_data, existing_codes,
        progress_path, save_path, logger
    )


def _download_daily_prices_tushare(
    pro, config, stock_codes, existing_data, existing_codes,
    progress_path, save_path, logger
) -> Optional[pd.DataFrame]:
    """通过 tushare 按交易日批量下载行情数据。"""
    try:
        # 获取交易日列表
        trading_dates = _get_trading_dates(pro, config, logger)
        if not trading_dates:
            return None

        # 确定需要下载的日期范围（如果有已有数据，只下载之后的日期）
        if existing_data is not None and len(existing_data) > 0:
            if isinstance(existing_data.index, pd.MultiIndex):
                last_date = existing_data.index.get_level_values("date").max()
            else:
                last_date = existing_data["date"].max()
            last_date_str = pd.Timestamp(last_date).strftime('%Y%m%d')
            trading_dates = [d for d in trading_dates if d > last_date_str]
            logger.info(f"已有数据截至 {last_date_str}，需补下载 {len(trading_dates)} 个交易日")

        if not trading_dates:
            return existing_data

        # 构建 ts_code 集合用于过滤
        ts_code_set = set(_code_to_ts_code(c) for c in stock_codes)

        all_data = []
        logger.info(f"[tushare] 开始下载 {len(trading_dates)} 个交易日的行情数据...")

        for trade_date in tqdm(trading_dates, desc="[tushare] 下载行情"):
            try:
                # 获取日线行情
                df = _ts_request(pro.daily, trade_date=trade_date)
                if df is None or len(df) == 0:
                    continue

                # 获取复权因子
                adj_df = _ts_request(pro.adj_factor, trade_date=trade_date)
                if adj_df is None or len(adj_df) == 0:
                    continue

                # 合并行情和复权因子
                df = df.merge(
                    adj_df[['ts_code', 'trade_date', 'adj_factor']],
                    on=['ts_code', 'trade_date'],
                    how='left'
                )

                # 过滤到目标股票
                df = df[df['ts_code'].isin(ts_code_set)]
                if len(df) == 0:
                    continue

                # 计算前复权价格
                # tushare 的 adj_factor 定义：复权价 = 原始价 × adj_factor / 最新adj_factor
                # 对于收益率计算，常数分母抵消，直接用 raw × adj_factor 即可
                for col in ['open', 'high', 'low', 'close']:
                    if col in df.columns:
                        df[col] = df[col] * df['adj_factor']

                # 重命名列
                df = df.rename(columns={
                    'ts_code': 'ts_code',
                    'trade_date': 'date',
                    'pre_close': 'pre_close',
                    'chg': 'chg',
                    'pct_chg': 'pct_change',
                    'vol': 'volume',  # tushare 用 vol 表示成交量
                })

                # 转换代码格式
                df['code'] = df['ts_code'].apply(_ts_code_to_code)
                df['date'] = pd.to_datetime(df['date'])

                # 选择需要的列
                keep_cols = ['date', 'code', 'open', 'high', 'low', 'close',
                             'pre_close', 'volume', 'amount', 'pct_change']
                available_cols = [c for c in keep_cols if c in df.columns]
                all_data.append(df[available_cols].copy())

            except Exception as e:
                logger.debug(f"  下载 {trade_date} 行情失败: {e}")
                continue

        if not all_data:
            return None

        new_df = pd.concat(all_data, ignore_index=True)

        # 合并已有数据
        if existing_data is not None and len(existing_data) > 0:
            existing_reset = existing_data.reset_index()
            result = pd.concat([existing_reset, new_df], ignore_index=True)
            result = result.drop_duplicates(subset=['date', 'code'])
        else:
            result = new_df

        result = result.set_index(['date', 'code']).sort_index()

        n_stocks = result.index.get_level_values('code').nunique()
        logger.info(f"[tushare] 行情数据: {len(result)} 条记录，覆盖 {n_stocks} 只股票")

        save_dataframe(result, save_path, logger)

        if n_stocks >= len(stock_codes) * 0.95:
            if progress_path.exists():
                progress_path.unlink()
            logger.info("行情数据下载完成")
        else:
            logger.warning(f"覆盖 {n_stocks}/{len(stock_codes)} 只股票，部分缺失")

        return result

    except Exception as e:
        logger.error(f"[tushare] 行情数据下载异常: {e}")
        logger.error(traceback.format_exc())
        return None


def _download_daily_prices_akshare(
    config, stock_codes, existing_data, existing_codes,
    progress_path, save_path, logger
) -> pd.DataFrame:
    """通过 akshare 逐只下载行情数据（回退方案）。"""
    # 确定需要下载的股票列表
    if existing_data is not None and len(existing_codes) > 0:
        codes_to_download = [c for c in stock_codes if c not in existing_codes]
        logger.info(f"[akshare] 数据不完整，需补下载 {len(codes_to_download)} 只")
    else:
        downloaded = set()
        if progress_path.exists():
            with open(progress_path, "r") as f:
                downloaded = set(line.strip() for line in f.readlines())
        codes_to_download = [c for c in stock_codes if c not in downloaded]
        if downloaded:
            logger.info(f"[akshare] 已有进度 {len(downloaded)} 只，剩余 {len(codes_to_download)} 只")

    start_date = config["data"]["start_date"].replace("-", "")
    end_date = config["data"]["end_date"].replace("-", "")

    new_data = []
    failed_codes = []

    logger.info(f"[akshare] 开始下载 {len(codes_to_download)} 只股票的日频行情...")

    for code in tqdm(codes_to_download, desc="[akshare] 下载行情"):
        try:
            df = _akshare_retry(
                ak.stock_zh_a_hist,
                symbol=code, period="daily",
                start_date=start_date, end_date=end_date, adjust="qfq"
            )

            if df is not None and len(df) > 0:
                df = df.rename(columns={
                    "日期": "date", "开盘": "open", "收盘": "close",
                    "最高": "high", "最低": "low", "成交量": "volume",
                    "成交额": "amount", "换手率": "turnover_rate",
                    "涨跌幅": "pct_change", "振幅": "amplitude",
                })
                df["code"] = code
                df["date"] = pd.to_datetime(df["date"])
                new_data.append(df)
                _safe_write_progress(progress_path, code)

            time.sleep(0.3)
        except Exception as e:
            logger.warning(f"[akshare] 下载 {code} 失败: {e}")
            failed_codes.append(code)
            time.sleep(1.0)

    if new_data:
        new_df = pd.concat(new_data, ignore_index=True)

        if existing_data is not None and len(existing_data) > 0:
            existing_reset = existing_data.reset_index()
            result = pd.concat([existing_reset, new_df], ignore_index=True)
        else:
            result = new_df

        result = result.drop_duplicates(subset=["date", "code"])
        result = result.set_index(["date", "code"]).sort_index()

        logger.info(f"[akshare] 行情数据: {len(result)} 条记录")
        if failed_codes:
            logger.warning(f"下载失败 {len(failed_codes)} 只: {failed_codes[:10]}")

        downloaded_count = len(existing_codes) + len(new_data)
        if len(failed_codes) == 0 and downloaded_count >= len(stock_codes):
            save_dataframe(result, save_path, logger)
            if progress_path.exists():
                progress_path.unlink()
        else:
            save_dataframe(result, save_path, logger)

        return result

    elif existing_data is not None and len(existing_data) > 0:
        logger.warning("本次未下载新数据，返回已有数据")
        return existing_data
    else:
        logger.error("没有成功下载任何数据！")
        return pd.DataFrame()


def _akshare_retry(func, max_retries: int = 3, sleep_seconds: float = 2.0, **kwargs):
    """akshare 带重试的调用。"""
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return func(**kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                time.sleep(sleep_seconds * (2 ** attempt))
            else:
                raise last_exception


# ============================================================
# 第三部分：估值指标下载
# ============================================================

def download_valuation_indicators(
    config: dict,
    stock_codes: List[str],
    logger
) -> pd.DataFrame:
    """
    下载估值指标：PE、PB、PS、总市值、换手率等。

    Tushare 策略：
        使用 daily_basic 接口按交易日批量获取。每次调用获取全市场当天的
        PE/PB/市值/换手率等指标，比逐只下载快约 300 倍。

    参数:
        config: 配置字典
        stock_codes: 股票代码列表
        logger: 日志记录器

    返回:
        DataFrame，MultiIndex 为 (date, code)
    """
    save_path = get_data_path(config, "raw") / "valuation.parquet"
    progress_path = get_data_path(config, "raw") / "_valuation_download_progress.txt"

    ensure_dir(get_data_path(config, "raw"))

    is_complete, existing_data, existing_codes = _check_data_completeness(
        save_path, stock_codes, logger, "估值数据"
    )
    if is_complete:
        return existing_data

    pro = _init_tushare(config, logger)

    if pro is not None:
        result = _download_valuation_tushare(
            pro, config, stock_codes, existing_data, existing_codes,
            progress_path, save_path, logger
        )
        if result is not None and len(result) > 0:
            return result
        logger.warning("[tushare] 估值数据下载失败，回退到 akshare")

    return _download_valuation_akshare(
        config, stock_codes, existing_data, existing_codes,
        progress_path, save_path, logger
    )


def _download_valuation_tushare(
    pro, config, stock_codes, existing_data, existing_codes,
    progress_path, save_path, logger
) -> Optional[pd.DataFrame]:
    """通过 tushare daily_basic 按交易日批量下载估值数据。"""
    try:
        # 获取交易日列表
        trading_dates = _get_trading_dates(pro, config, logger)
        if not trading_dates:
            return None

        # 如果有已有数据，只下载之后的日期
        if existing_data is not None and len(existing_data) > 0:
            if isinstance(existing_data.index, pd.MultiIndex):
                last_date = existing_data.index.get_level_values("date").max()
            else:
                last_date = existing_data["date"].max()
            last_date_str = pd.Timestamp(last_date).strftime('%Y%m%d')
            trading_dates = [d for d in trading_dates if d > last_date_str]
            logger.info(f"已有估值数据截至 {last_date_str}，需补下载 {len(trading_dates)} 个交易日")

        if not trading_dates:
            return existing_data

        ts_code_set = set(_code_to_ts_code(c) for c in stock_codes)

        all_data = []
        logger.info(f"[tushare] 开始下载 {len(trading_dates)} 个交易日的估值指标...")

        for trade_date in tqdm(trading_dates, desc="[tushare] 下载估值"):
            try:
                # daily_basic 需要 2000 积分
                df = _ts_request(
                    pro.daily_basic,
                    trade_date=trade_date,
                    fields='ts_code,trade_date,turnover_rate,pe,pe_ttm,pb,ps,ps_ttm,'
                           'dv_ratio,dv_ttm,total_share,float_share,total_mv,circ_mv'
                )

                if df is None or len(df) == 0:
                    continue

                # 过滤到目标股票
                df = df[df['ts_code'].isin(ts_code_set)]
                if len(df) == 0:
                    continue

                # 重命名列
                df = df.rename(columns={
                    'trade_date': 'date',
                    'pe': 'pe',
                    'pe_ttm': 'pe_ttm',
                    'pb': 'pb',
                    'ps': 'ps',
                    'ps_ttm': 'ps_ttm',
                    'dv_ratio': 'dv_ratio',
                    'dv_ttm': 'dv_ttm',
                    'total_mv': 'total_mv',
                    'circ_mv': 'circ_mv',
                    'turnover_rate': 'turnover_rate',
                })

                df['code'] = df['ts_code'].apply(_ts_code_to_code)
                df['date'] = pd.to_datetime(df['date'])

                keep_cols = ['date', 'code', 'pe_ttm', 'pb', 'ps_ttm',
                             'total_mv', 'dv_ratio', 'turnover_rate']
                available_cols = [c for c in keep_cols if c in df.columns]
                all_data.append(df[available_cols].copy())

            except Exception as e:
                logger.debug(f"  下载 {trade_date} 估值失败: {e}")
                continue

        if not all_data:
            return None

        new_df = pd.concat(all_data, ignore_index=True)

        if existing_data is not None and len(existing_data) > 0:
            existing_reset = existing_data.reset_index()
            result = pd.concat([existing_reset, new_df], ignore_index=True)
            result = result.drop_duplicates(subset=['date', 'code'])
        else:
            result = new_df

        result = result.set_index(['date', 'code']).sort_index()

        n_stocks = result.index.get_level_values('code').nunique()
        logger.info(f"[tushare] 估值数据: {len(result)} 条记录，覆盖 {n_stocks} 只股票")

        save_dataframe(result, save_path, logger)

        if n_stocks >= len(stock_codes) * 0.95:
            if progress_path.exists():
                progress_path.unlink()

        return result

    except Exception as e:
        logger.error(f"[tushare] 估值数据下载异常: {e}")
        return None


def _download_valuation_akshare(
    config, stock_codes, existing_data, existing_codes,
    progress_path, save_path, logger
) -> pd.DataFrame:
    """通过 akshare 逐只下载估值数据（回退方案）。"""
    if existing_data is not None and len(existing_codes) > 0:
        codes_to_download = [c for c in stock_codes if c not in existing_codes]
        logger.info(f"[akshare] 估值数据不完整，需补下载 {len(codes_to_download)} 只")
    else:
        downloaded = set()
        if progress_path.exists():
            with open(progress_path, "r") as f:
                downloaded = set(line.strip() for line in f.readlines())
        codes_to_download = [c for c in stock_codes if c not in downloaded]

    all_data = []
    failed_codes = []

    indicators = {
        "总市值": "total_mv",
        "市盈率(TTM)": "pe_ttm",
        "市净率": "pb",
        "市销率": "ps_ttm",
    }

    logger.info(f"[akshare] 开始下载 {len(codes_to_download)} 只股票的估值指标...")

    for code in tqdm(codes_to_download, desc="[akshare] 下载估值"):
        try:
            pure_code = code.split('.')[0] if '.' in code else code
            pure_code = pure_code.replace('SH', '').replace('SZ', '')

            stock_data = {}
            dates_set = None

            for indicator_name, col_name in indicators.items():
                try:
                    df = _akshare_retry(
                        ak.stock_zh_valuation_baidu,
                        symbol=pure_code,
                        indicator=indicator_name,
                        period="近三年"
                    )
                    if df is not None and len(df) > 0:
                        df["date"] = pd.to_datetime(df["date"])
                        if dates_set is None:
                            dates_set = set(df["date"])
                        else:
                            dates_set = dates_set & set(df["date"])
                        stock_data[col_name] = df.set_index("date")["value"]
                    time.sleep(0.2)
                except Exception:
                    continue

            if dates_set and len(dates_set) > 0:
                merged = pd.DataFrame({"date": sorted(dates_set), "code": code})
                merged["date"] = pd.to_datetime(merged["date"])
                for col_name, series in stock_data.items():
                    if isinstance(series, pd.Series):
                        series.index = pd.to_datetime(series.index)
                        merged[col_name] = merged["date"].map(series)

                start = pd.to_datetime(config["data"]["start_date"])
                end = pd.to_datetime(config["data"]["end_date"])
                merged = merged[(merged["date"] >= start) & (merged["date"] <= end)]

                if "total_mv" in merged.columns and merged["total_mv"].notna().sum() > 0:
                    all_data.append(merged)
                    _safe_write_progress(progress_path, code)

            time.sleep(0.3)
        except Exception as e:
            logger.warning(f"[akshare] 下载 {code} 估值失败: {e}")
            failed_codes.append(code)
            time.sleep(1.0)

    if all_data:
        new_df = pd.concat(all_data, ignore_index=True)
        if existing_data is not None and len(existing_data) > 0:
            existing_reset = existing_data.reset_index()
            result = pd.concat([existing_reset, new_df], ignore_index=True)
        else:
            result = new_df
        result = result.drop_duplicates(subset=["date", "code"])
        result = result.set_index(["date", "code"]).sort_index()

        logger.info(f"[akshare] 估值数据: {len(result)} 条记录")
        downloaded_count = len(existing_codes) + len(all_data)
        if len(failed_codes) == 0 and downloaded_count >= len(stock_codes):
            save_dataframe(result, save_path, logger)
            if progress_path.exists():
                progress_path.unlink()
        else:
            save_dataframe(result, save_path, logger)
        return result

    elif existing_data is not None and len(existing_data) > 0:
        return existing_data
    else:
        logger.error("估值数据下载失败！")
        return pd.DataFrame()


# ============================================================
# 第四部分：财务指标下载
# ============================================================

def download_financial_data(
    config: dict,
    stock_codes: List[str],
    logger
) -> pd.DataFrame:
    """
    下载财务指标：ROE、ROA、毛利率、净利率、负债率、流动比率、营收增速。

    Tushare 策略：
        使用 fina_indicator 接口逐只获取（该接口不支持按日期批量获取全市场）。
        但 tushare 的财务数据字段更规范，不需要手动映射中文列名。

    参数:
        config: 配置字典
        stock_codes: 股票代码列表
        logger: 日志记录器

    返回:
        DataFrame，包含财务指标
    """
    save_path = get_data_path(config, "raw") / "financial_data.parquet"
    progress_path = get_data_path(config, "raw") / "_financial_download_progress.txt"

    ensure_dir(get_data_path(config, "raw"))

    is_complete, existing_data, existing_codes = _check_data_completeness(
        save_path, stock_codes, logger, "财务数据"
    )
    if is_complete:
        return existing_data

    pro = _init_tushare(config, logger)

    if pro is not None:
        result = _download_financial_tushare(
            pro, config, stock_codes, existing_data, existing_codes,
            progress_path, save_path, logger
        )
        if result is not None and len(result) > 0:
            return result
        logger.warning("[tushare] 财务数据下载失败，回退到 akshare")

    return _download_financial_akshare(
        config, stock_codes, existing_data, existing_codes,
        progress_path, save_path, logger
    )


def _download_financial_tushare(
    pro, config, stock_codes, existing_data, existing_codes,
    progress_path, save_path, logger
) -> Optional[pd.DataFrame]:
    """通过 tushare fina_indicator 逐只下载财务指标。"""
    try:
        if existing_data is not None and len(existing_codes) > 0:
            codes_to_download = [c for c in stock_codes if c not in existing_codes]
        else:
            downloaded = set()
            if progress_path.exists():
                with open(progress_path, "r") as f:
                    downloaded = set(line.strip() for line in f.readlines())
            codes_to_download = [c for c in stock_codes if c not in downloaded]

        if not codes_to_download:
            return existing_data

        delay_config = config["data"]["financial_report_delay"]
        all_data = []
        failed_codes = []

        logger.info(f"[tushare] 开始下载 {len(codes_to_download)} 只股票的财务指标...")

        for code in tqdm(codes_to_download, desc="[tushare] 下载财务"):
            ts_code = _code_to_ts_code(code)
            try:
                # fina_indicator 需要 2000 积分
                df = _ts_request(
                    pro.fina_indicator,
                    ts_code=ts_code,
                    start_date=config["data"]["start_date"].replace("-", ""),
                    end_date=config["data"]["end_date"].replace("-", ""),
                    fields=(
                        'ts_code,ann_date,end_date,report_type,comp_type,update_flag,'
                        'roe_dt,roa,grossprofit_margin,netprofit_margin,'
                        'debt_to_assets,current_ratio,or_yoy,'
                        'ocfps,roe_yoy,dt_netprofit_yoy'
                    )
                )

                if df is not None and len(df) > 0:
                    df = df.rename(columns={
                        'ann_date': 'publish_date',
                        'end_date': 'report_date',
                        'roe_dt': 'roe',         # 扣非 ROE
                        'roa': 'roa',
                        'grossprofit_margin': 'gross_margin',
                        'netprofit_margin': 'net_margin',
                        'debt_to_assets': 'debt_ratio',
                        'current_ratio': 'current_ratio',
                        'or_yoy': 'revenue_growth',  # 营收同比增速
                        'ocfps': 'ocfps',              # 每股经营现金流
                        'roe_yoy': 'roe_yoy',          # ROE 同比变化
                        'dt_netprofit_yoy': 'dt_netprofit_yoy',  # 扣非净利润同比
                    })

                    df['code'] = code
                    df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')

                    # 同一 (code, report_date) 可能因 report_type 不同而重复，去重保留最新
                    df = df.sort_values('report_date').drop_duplicates(
                        subset=['code', 'report_date'], keep='last'
                    )

                    all_data.append(df)
                    _safe_write_progress(progress_path, code)

            except Exception as e:
                logger.warning(f"[tushare] 下载 {code} 财务数据失败: {e}")
                failed_codes.append(code)

        if all_data:
            new_df = pd.concat(all_data, ignore_index=True)
            if existing_data is not None and len(existing_data) > 0 :
                result = pd.concat([existing_data, new_df], ignore_index=True)
                result = result.drop_duplicates()
            else :
                result = new_df

            logger.info(f"[tushare] 财务数据: {len(result)} 条记录")
            if failed_codes:
                logger.warning(f"下载失败 {len(failed_codes)} 只: {failed_codes[:10]}")

            downloaded_count = len(existing_codes) + len(all_data)
            if len(failed_codes) == 0 and downloaded_count >= len(stock_codes):
                save_dataframe(result, save_path, logger)
                if progress_path.exists():
                    progress_path.unlink()
            else:
                save_dataframe(result, save_path, logger)

            return result

        return None

    except Exception as e:
        logger.error(f"[tushare] 财务数据下载异常: {e}")
        return None


def _download_financial_akshare(
    config, stock_codes, existing_data, existing_codes,
    progress_path, save_path, logger
) -> pd.DataFrame:
    """通过 akshare 逐只下载财务数据（回退方案）。"""
    if existing_data is not None and len(existing_codes) > 0:
        codes_to_download = [c for c in stock_codes if c not in existing_codes]
    else:
        downloaded = set()
        if progress_path.exists():
            with open(progress_path, "r") as f:
                downloaded = set(line.strip() for line in f.readlines())
        codes_to_download = [c for c in stock_codes if c not in downloaded]

    if not codes_to_download:
        return existing_data if existing_data is not None else pd.DataFrame()

    delay_config = config["data"]["financial_report_delay"]
    all_data = []
    failed_codes = []

    logger.info(f"[akshare] 开始下载 {len(codes_to_download)} 只股票的财务指标...")

    for code in tqdm(codes_to_download, desc="[akshare] 下载财务"):
        try:
            df = _akshare_retry(
                ak.stock_financial_analysis_indicator,
                symbol=code
            )

            if df is not None and len(df) > 0:
                col_map = {}
                for col in df.columns:
                    if "净资产收益率" in col or "roe" in col.lower():
                        col_map[col] = "roe"
                    elif "总资产报酬率" in col or "roa" in col.lower():
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

                date_col = None
                for col in df.columns:
                    if "日期" in col or "date" in col.lower() or "报告期" in col:
                        date_col = col
                        break

                if date_col:
                    df["report_date"] = pd.to_datetime(df[date_col], errors="coerce")
                    df["publish_date"] = df["report_date"].apply(
                        lambda d: _estimate_publish_date(d, delay_config) if pd.notna(d) else pd.NaT
                    )
                    all_data.append(df)

            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"[akshare] 下载 {code} 财务数据失败: {e}")
            failed_codes.append(code)

    if all_data:
        new_df = pd.concat(all_data, ignore_index=True)
        if existing_data is not None and len(existing_data) > 0:
            result = pd.concat([existing_data, new_df], ignore_index=True)
            result = result.drop_duplicates()
        else:
            result = new_df

        logger.info(f"[akshare] 财务数据: {len(result)} 条记录")
        downloaded_count = len(existing_codes) + len(all_data)
        if len(failed_codes) == 0 and downloaded_count >= len(stock_codes):
            save_dataframe(result, save_path, logger)
            if progress_path.exists():
                progress_path.unlink()
        else:
            save_dataframe(result, save_path, logger)
        return result

    elif existing_data is not None and len(existing_data) > 0:
        return existing_data
    else:
        logger.error("财务数据下载失败！")
        return pd.DataFrame()


# ============================================================
# 第五部分：指数行情下载
# ============================================================

def download_index_data(config: dict, logger) -> pd.DataFrame:
    """
    下载基准指数的日线行情。

    参数:
        config: 配置字典
        logger: 日志记录器

    返回:
        DataFrame，index 为日期，包含 open/high/low/close/volume 列
    """
    save_path = get_data_path(config, "raw") / "index_daily.parquet"

    if save_path.exists():
        logger.info(f"从本地加载指数数据: {save_path}")
        return load_dataframe(save_path, logger)

    ensure_dir(get_data_path(config, "raw"))

    universe = config["data"]["universe"]
    ts_index_code = "000300.SH" if universe == "hs300" else "000905.SH"
    ak_index_code = "000300" if universe == "hs300" else "000905"
    index_name = "沪深300" if universe == "hs300" else "中证500"

    pro = _init_tushare(config, logger)

    if pro is not None:
        try:
            logger.info(f"[tushare] 下载 {index_name} 指数行情...")
            df = _ts_request(
                pro.index_daily,
                ts_code=ts_index_code,
                start_date=config["data"]["start_date"].replace("-", ""),
                end_date=config["data"]["end_date"].replace("-", "")
            )

            if df is not None and len(df) > 0:
                df = df.rename(columns={
                    'trade_date': 'date',
                })
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()

                # 多取一些数据用于因子计算
                start = pd.to_datetime(config["data"]["start_date"]) - pd.Timedelta(days=400)
                if df.index.min() > start:
                    # 扩展时间范围重新下载
                    df = _ts_request(
                        pro.index_daily,
                        ts_code=ts_index_code,
                        start_date=start.strftime('%Y%m%d'),
                        end_date=config["data"]["end_date"].replace("-", "")
                    )
                    df = df.rename(columns={'trade_date': 'date'})
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date').sort_index()

                logger.info(f"[tushare] 指数数据: {len(df)} 个交易日")
                save_dataframe(df, save_path, logger)
                return df

        except Exception as e:
            logger.warning(f"[tushare] 下载指数数据失败: {e}")

    # 回退到 akshare
    logger.info(f"[akshare] 下载 {index_name} 指数行情...")
    try:
        df = _akshare_retry(
            ak.stock_zh_index_daily,
            symbol=f"sh{ak_index_code}"
        )

        if df is not None and len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

            start = pd.to_datetime(config["data"]["start_date"]) - pd.Timedelta(days=400)
            end = pd.to_datetime(config["data"]["end_date"])
            df = df[(df.index >= start) & (df.index <= end)]

            logger.info(f"[akshare] 指数数据: {len(df)} 个交易日")
            save_dataframe(df, save_path, logger)
            return df

    except Exception as e:
        logger.error(f"下载指数数据失败: {e}")
        raise

    return pd.DataFrame()


# ============================================================
# 第六部分：行业分类获取
# ============================================================

def download_industry_classification(config: dict, stock_codes: List[str], logger) -> pd.DataFrame:
    """
    获取申万一级行业分类。

    Tushare 的申万行业分类接口（index_classify / index_member）需要较高积分（5000+），
    因此默认直接回退到 akshare。如果 tushare 积分足够，可尝试 tushare 源。

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

    ensure_dir(get_data_path(config, "raw"))

    pro = _init_tushare(config, logger)

    # 尝试 tushare 的申万行业分类（需要 5000+ 积分）
    if pro is not None:
        try:
            logger.info("[tushare] 尝试获取申万行业分类...")

            # 获取申万一级行业列表
            classify_df = _ts_request(
                pro.index_classify,
                level='L1',
                src='SW2021'
            )

            if classify_df is not None and len(classify_df) > 0:
                industry_list = []

                # 逐个行业获取成分股
                for _, row in tqdm(classify_df.iterrows(), total=len(classify_df),
                                   desc="[tushare] 获取行业成分"):
                    index_code = row.get('index_code', '')
                    industry_name = row.get('industry_name', '')

                    try:
                        member_df = _ts_request(
                            pro.index_member,
                            index_code=index_code
                        )
                        if member_df is not None and len(member_df) > 0:
                            for _, m_row in member_df.iterrows():
                                con_code = m_row.get('con_code', '')
                                code = _ts_code_to_code(con_code)
                                if code in stock_codes:
                                    industry_list.append({
                                        "code": code,
                                        "industry": industry_name
                                    })
                    except Exception:
                        continue

                    time.sleep(0.3)

                if industry_list:
                    result = pd.DataFrame(industry_list).drop_duplicates(subset=["code"])
                    logger.info(f"[tushare] 获取到 {len(result)} 只股票的行业分类")
                    save_dataframe(result, save_path, logger)
                    return result

        except Exception as e:
            logger.warning(f"[tushare] 行业分类获取失败: {e}，回退到 akshare")

    # 回退到 akshare（默认路径）
    logger.info("[akshare] 获取申万一级行业分类...")
    try:
        industry_list = []
        board_df = _akshare_retry(ak.stock_board_industry_name_em, board_type="industry")

        if board_df is not None and len(board_df) > 0:
            for _, row in tqdm(board_df.iterrows(), total=len(board_df),
                               desc="[akshare] 获取行业成分"):
                board_name = row.get("板块名称", row.iloc[1] if len(row) > 1 else None)
                if board_name is None:
                    continue

                try:
                    cons_df = _akshare_retry(
                        ak.stock_board_industry_cons_em,
                        symbol=board_name
                    )
                    if cons_df is not None and len(cons_df) > 0:
                        code_col = None
                        for col in cons_df.columns:
                            if "代码" in col or "code" in col.lower():
                                code_col = col
                                break
                        if code_col is None:
                            code_col = cons_df.columns[1]

                        codes = cons_df[code_col].astype(str).str.zfill(6).tolist()
                        for code in codes:
                            if code in stock_codes:
                                industry_list.append({"code": code, "industry": board_name})

                    time.sleep(0.2)
                except Exception:
                    continue

        if industry_list:
            result = pd.DataFrame(industry_list).drop_duplicates(subset=["code"])
            logger.info(f"[akshare] 获取到 {len(result)} 只股票的行业分类，"
                        f"覆盖 {result['industry'].nunique()} 个行业")
            save_dataframe(result, save_path, logger)
            return result
        else:
            logger.warning("行业分类获取失败，将使用空数据")
            return pd.DataFrame(columns=["code", "industry"])

    except Exception as e:
        logger.error(f"获取行业分类失败: {e}")
        return pd.DataFrame(columns=["code", "industry"])


# ============================================================
# 第七部分：数据清洗
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

    required_files = ["daily_prices.parquet", "valuation.parquet", "index_daily.parquet"]
    for f in required_files:
        if not (raw_path / f).exists():
            logger.error(f"必要的数据文件不存在: {raw_path / f}")
            logger.error("请先运行: python src/data_loader_tushare.py --config config/config.yaml")
            raise FileNotFoundError(f"缺少数据文件: {f}")

    logger.info("加载原始数据...")
    prices = load_dataframe(raw_path / "daily_prices.parquet", logger)
    valuation = load_dataframe(raw_path / "valuation.parquet", logger)
    index_data = load_dataframe(raw_path / "index_daily.parquet", logger)

    financial_data = None
    if (raw_path / "financial_data.parquet").exists():
        financial_data = load_dataframe(raw_path / "financial_data.parquet", logger)

    industry = None
    if (raw_path / "industry_classification.parquet").exists():
        industry = load_dataframe(raw_path / "industry_classification.parquet", logger)

    # 剔除上市不足 N 日的股票
    logger.info(f"剔除上市不足 {min_list_days} 日的股票...")
    stock_trade_days = prices.groupby(level="code").size()
    valid_stocks = stock_trade_days[stock_trade_days >= min_list_days].index
    prices = prices.loc[prices.index.get_level_values("code").isin(valid_stocks)]
    valuation = valuation.loc[valuation.index.get_level_values("code").isin(valid_stocks)]
    logger.info(f"保留 {len(valid_stocks)} 只股票"
                f"（剔除 {stock_trade_days.count() - len(valid_stocks)} 只）")

    # 剔除停牌日
    logger.info("剔除停牌日（成交量为 0 的记录）...")
    if "volume" in prices.columns:
        suspended = prices["volume"] == 0
        n_suspended = suspended.sum()
        prices = prices[~suspended]
        logger.info(f"剔除 {n_suspended} 条停牌记录")

    # 合并估值数据到行情数据
    logger.info("合并行情与估值数据...")
    merged = prices.join(valuation, how="left")

    # 保存清洗后的数据
    ensure_dir(proc_path)
    logger.info("保存清洗后的数据...")
    save_dataframe(merged, proc_path / "market_data.parquet", logger)
    save_dataframe(index_data, proc_path / "index_data.parquet", logger)

    if financial_data is not None:
        save_dataframe(financial_data, proc_path / "financial_data.parquet", logger)
    if industry is not None:
        save_dataframe(industry, proc_path / "industry.parquet", logger)

    result = {"market_data": merged, "index_data": index_data}
    if financial_data is not None:
        result["financial_data"] = financial_data
    if industry is not None:
        result["industry"] = industry

    return result


# ============================================================
# 第七点三部分：北向资金持股与分析师评级下载
# ============================================================

def download_northbound_data(
    config: dict,
    stock_codes: List[str],
    logger
) -> pd.DataFrame:
    """
    下载北向资金（沪深港通）持股数据。

    使用 hk_hold 接口按日期范围获取，提取 ratio（持股占比）字段。
    数据为日频，保存为 (date, code) MultiIndex 格式。

    参数:
        config: 配置字典
        stock_codes: 目标股票代码列表
        logger: 日志记录器

    返回:
        DataFrame，MultiIndex 为 (date, code)，包含 northbound_ratio 列
    """
    save_path = get_data_path(config, "raw") / "northbound_data.parquet"

    if save_path.exists():
        logger.info(f"从本地加载北向资金数据: {save_path}")
        return load_dataframe(save_path, logger)

    ensure_dir(get_data_path(config, "raw"))

    pro = _init_tushare(config, logger)
    if pro is None:
        logger.warning("Tushare 不可用，无法获取北向资金数据")
        return pd.DataFrame()

    try:
        ts_code_set = set(_code_to_ts_code(c) for c in stock_codes)
        start_date = config["data"]["start_date"].replace("-", "")
        end_date = config["data"]["end_date"].replace("-", "")

        # 按 2 周分块（~300 只股票 × 14 天 ≈ 4200，接近 API 上限）
        date_chunks = pd.date_range(start=start_date, end=end_date, freq="2W")
        date_ranges = []
        for i in range(len(date_chunks)):
            chunk_start = date_chunks[i].strftime("%Y%m%d")
            chunk_end = (
                (date_chunks[i + 1] - pd.Timedelta(days=1)).strftime("%Y%m%d")
                if i < len(date_chunks) - 1
                else end_date
            )
            date_ranges.append((chunk_start, chunk_end))

        all_data = []
        logger.info(f"[tushare] 开始下载北向资金持股数据 ({len(date_ranges)} 个时间段)...")

        for s_date, e_date in tqdm(date_ranges, desc="[tushare] 下载北向资金"):
            for exchange in ["SH", "SZ"]:
                try:
                    df = _ts_request(
                        pro.hk_hold,
                        exchange=exchange,
                        start_date=s_date,
                        end_date=e_date
                    )
                    if df is None or len(df) == 0:
                        continue

                    df = df[df['ts_code'].isin(ts_code_set)]
                    if len(df) == 0:
                        continue

                    df = df[['ts_code', 'trade_date', 'ratio']].copy()
                    df['code'] = df['ts_code'].apply(_ts_code_to_code)
                    df['date'] = pd.to_datetime(df['trade_date'])
                    df = df.rename(columns={'ratio': 'northbound_ratio'})
                    all_data.append(df[['date', 'code', 'northbound_ratio']])

                except Exception as e:
                    logger.debug(f"  下载北向资金 {exchange} {s_date}~{e_date} 失败: {e}")
                    continue

        if not all_data:
            logger.warning("未获取到北向资金数据")
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=['date', 'code'])
        result = result.set_index(['date', 'code']).sort_index()

        n_stocks = result.index.get_level_values('code').nunique()
        n_dates = result.index.get_level_values('date').nunique()
        logger.info(
            f"[tushare] 北向资金数据: {len(result)} 条，"
            f"覆盖 {n_stocks} 只股票，{n_dates} 个交易日"
        )

        save_dataframe(result, save_path, logger)
        return result

    except Exception as e:
        logger.error(f"[tushare] 北向资金数据下载异常: {e}")
        return pd.DataFrame()


def download_consensus_ratings(
    config: dict,
    stock_codes: List[str],
    logger
) -> pd.DataFrame:
    """
    下载分析师一致评级数据。

    使用 report_rc 接口获取券商研报评级，将评级文本映射为数值分数后
    按 (stock, date) 取均值，得到每日一致评级分数。

    注意:
        report_rc 接口限制 10 次/天，72 个月度分块需约 7 天完成。
        支持断点续传：已下载的月份会记录在进度文件中，下次运行时跳过。

    评级映射:
        买入/强烈推荐/推荐/BUY = 5
        增持/优于大市/OVERWEIGHT = 4
        中性/观望 = 3
        减持 = 2
        卖出 = 1

    参数:
        config: 配置字典
        stock_codes: 目标股票代码列表
        logger: 日志记录器

    返回:
        DataFrame，MultiIndex 为 (date, code)，包含 consensus_rating 列
    """
    save_path = get_data_path(config, "raw") / "consensus_ratings.parquet"
    progress_path = get_data_path(config, "raw") / "_consensus_download_progress.txt"

    if save_path.exists():
        existing = load_dataframe(save_path, logger)
        if existing is not None and len(existing) > 0:
            # 检查是否已覆盖全部时间范围
            n_dates = existing.index.get_level_values("date").nunique()
            if n_dates > 100:
                logger.info(f"从本地加载分析师评级数据: {save_path} ({len(existing)} 条)")
                return existing
            else:
                logger.info(f"本地数据不完整（{n_dates} 天），尝试续传...")

    ensure_dir(get_data_path(config, "raw"))

    pro = _init_tushare(config, logger)
    if pro is None:
        logger.warning("Tushare 不可用，无法获取分析师评级数据")
        return pd.DataFrame()

    try:
        ts_code_set = set(_code_to_ts_code(c) for c in stock_codes)
        start_date = config["data"]["start_date"].replace("-", "")
        end_date = config["data"]["end_date"].replace("-", "")

        # 评级文本 → 数值映射
        rating_map = {
            '买入': 5, '强烈推荐': 5, '推荐': 5, 'BUY': 5,
            '增持': 4, '优于大市': 4, 'OVERWEIGHT': 4,
            '中性': 3, '观望': 3, 'NEUTRAL': 3, 'HOLD': 3,
            '减持': 2, 'REDUCE': 2,
            '卖出': 1, 'SELL': 1,
        }

        # 按月分块
        date_chunks = pd.date_range(start=start_date, end=end_date, freq="MS")
        date_ranges = []
        for i in range(len(date_chunks)):
            chunk_start = date_chunks[i].strftime("%Y%m%d")
            chunk_end = (
                (date_chunks[i + 1] - pd.Timedelta(days=1)).strftime("%Y%m%d")
                if i < len(date_chunks) - 1
                else end_date
            )
            date_ranges.append((chunk_start, chunk_end))

        # 读取进度（已完成的月份）
        downloaded_chunks = set()
        if progress_path.exists():
            with open(progress_path, "r") as f:
                downloaded_chunks = set(line.strip() for line in f if line.strip())

        # 加载已有数据（用于增量合并）
        existing_data = None
        if save_path.exists():
            try:
                existing_data = load_dataframe(save_path, logger)
            except Exception:
                pass

        remaining = [(s, e) for s, e in date_ranges if f"{s}_{e}" not in downloaded_chunks]
        if downloaded_chunks:
            logger.info(f"已完成 {len(downloaded_chunks)} 个月，剩余 {len(remaining)} 个月")

        if not remaining:
            if existing_data is not None and len(existing_data) > 0:
                return existing_data

        # report_rc 限制 10 次/天
        DAILY_LIMIT = 9  # 留 1 次安全余量
        daily_calls = 0

        all_data = []
        if existing_data is not None and len(existing_data) > 0:
            all_data.append(existing_data.reset_index())

        rate_limited = False
        logger.info(f"[tushare] 开始下载分析师评级数据 ({len(remaining)} 个待下载时间段)...")
        logger.info(f"  注意: report_rc 限制 10 次/天，本次最多执行 {DAILY_LIMIT} 次调用")

        for s_date, e_date in tqdm(remaining, desc="[tushare] 下载评级"):
            if daily_calls >= DAILY_LIMIT:
                rate_limited = True
                logger.warning(
                    f"  已达今日调用上限 ({daily_calls} 次)，"
                    f"剩余 {len(remaining) - daily_calls} 个月度分块将在下次运行时续传"
                )
                break

            try:
                df = _ts_request(
                    pro.report_rc,
                    start_date=s_date,
                    end_date=e_date,
                    fields='ts_code,report_date,rating'
                )
                daily_calls += 1

                if df is None or len(df) == 0:
                    _safe_write_progress(progress_path, f"{s_date}_{e_date}")
                    continue

                # 映射评级为数值
                df['rating_score'] = df['rating'].map(rating_map)
                df = df.dropna(subset=['rating_score'])

                # 过滤目标股票
                df = df[df['ts_code'].isin(ts_code_set)]
                if len(df) == 0:
                    _safe_write_progress(progress_path, f"{s_date}_{e_date}")
                    continue

                df['code'] = df['ts_code'].apply(_ts_code_to_code)
                df['date'] = pd.to_datetime(df['report_date'])

                # 按 (stock, date) 取均值 → 一致评级
                consensus = df.groupby(['code', 'date'])['rating_score'].mean().reset_index()
                consensus = consensus.rename(columns={'rating_score': 'consensus_rating'})
                all_data.append(consensus)
                _safe_write_progress(progress_path, f"{s_date}_{e_date}")

            except Exception as e:
                err_msg = str(e).lower()
                if 'freq' in err_msg or 'limit' in err_msg or '频次' in err_msg or '超限' in err_msg:
                    logger.warning(f"  触发频率限制，今日下载停止。已完成 {daily_calls} 次调用")
                    rate_limited = True
                    break
                logger.debug(f"  下载评级 {s_date}~{e_date} 失败: {e}")
                continue

        if not all_data:
            logger.warning("未获取到分析师评级数据")
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=['date', 'code'])
        result = result.set_index(['date', 'code']).sort_index()

        n_stocks = result.index.get_level_values('code').nunique()
        n_dates = result.index.get_level_values('date').nunique()
        logger.info(
            f"[tushare] 分析师评级数据: {len(result)} 条，"
            f"覆盖 {n_stocks} 只股票，{n_dates} 个交易日"
        )

        if not rate_limited:
            # 全部下载完成，清理进度文件
            save_dataframe(result, save_path, logger)
            if progress_path.exists():
                progress_path.unlink()
            logger.info("分析师评级数据下载完成")
        else:
            # 部分完成，保存中间结果
            save_dataframe(result, save_path, logger)
            logger.info(f"已保存中间结果（{len(result)} 条），请明天继续运行以补全数据")

        return result

    except Exception as e:
        logger.error(f"[tushare] 分析师评级数据下载异常: {e}")
        return pd.DataFrame()


# ============================================================
# 第七点五部分：流水线入口（供 pipeline.py 调用）
# ============================================================

def run_data_pipeline(config: dict, logger, skip_download: bool = False):
    """
    数据获取与清洗的完整流程（供 pipeline.py 统一调用）。

    参数:
        config: 配置字典
        logger: 日志记录器
        skip_download: 是否跳过下载，只做数据清洗
    """
    if not skip_download:
        constituents = get_index_constituents(config, logger)
        stock_codes = constituents["code"].tolist()
        download_daily_prices(config, stock_codes, logger)
        download_valuation_indicators(config, stock_codes, logger)
        download_financial_data(config, stock_codes, logger)
        download_index_data(config, logger)
        download_industry_classification(config, stock_codes, logger)
        download_northbound_data(config, stock_codes, logger)
        download_consensus_ratings(config, stock_codes, logger)

    logger.info("开始数据清洗...")
    clean_data(config, logger)
    logger.info("数据获取与清洗完成！")


# ============================================================
# 第八部分：主函数（命令行入口）
# ============================================================

def main():
    """
    命令行入口：执行完整的数据获取与清洗流程。

    用法:
        python src/data_loader_tushare.py --config config/config.yaml
        python src/data_loader_tushare.py --config config/config.yaml --skip-download
    """
    parser = argparse.ArgumentParser(description="数据获取与清洗（Tushare 版）")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="配置文件路径")
    parser.add_argument("--skip-download", action="store_true",
                        help="跳过下载，只做数据清洗")
    parser.add_argument("--run_id", type=int, default=None,
                        help="运行序号（默认自动递增）")
    args = parser.parse_args()

    config = load_config(args.config)
    run_id = init_run(config, args.run_id)
    logger = setup_logger("data_loader_tushare", config.get("log_level", "INFO"))

    logger.info(f"运行 ID: {run_id}")
    logger.info("=" * 60)
    logger.info("多因子选股项目 - 数据获取与清洗（Tushare 版）")
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

        # 步骤 7：下载北向资金持股数据
        download_northbound_data(config, stock_codes, logger)

        # 步骤 8：下载分析师一致评级
        # download_consensus_ratings(config, stock_codes, logger)

    # 步骤 9：数据清洗
    logger.info("开始数据清洗...")
    cleaned = clean_data(config, logger)

    logger.info("=" * 60)
    logger.info("数据获取与清洗全部完成！")
    logger.info(f"处理后数据保存在: {get_data_path(config, 'processed')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
