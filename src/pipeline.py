"""
pipeline.py - 统一流水线入口
================================
一键运行完整流程：数据获取 → 因子计算 → 标签构建 → 预处理 → 时间划分 → 模型训练 → 回测 → 评估 → 归因分析。

支持参数:
    --config    配置文件路径（默认 config/config.yaml）
    --run_id    手动指定运行序号（默认自动递增）
    --skip      跳过指定步骤（逗号分隔，如 --skip data,analysis）
    --only      只运行指定步骤（逗号分隔，如 --only models,backtest,evaluation）

运行方式:
    # 跑全流程
    python src/pipeline.py

    # 跳过数据下载（数据已存在时）
    python src/pipeline.py --skip data

    # 只跑模型训练 + 回测 + 评估
    python src/pipeline.py --only models,backtest,evaluation

    # 手动指定运行序号
    python src/pipeline.py --run_id 5
"""

import argparse
import time
import sys
import os

# 确保 src/ 在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_config, setup_logger, init_run, get_run_dir

# 所有可用步骤（按执行顺序）
ALL_STEPS = [
    "data",         # 数据获取与清洗
    "features",     # 因子计算
    "labels",       # 标签构建
    "preprocess",   # 因子预处理
    "split",        # 时间序列划分
    "models",       # 模型训练与预测
    "backtest",     # 回测
    "evaluation",   # 评估
    "analysis",     # 归因分析
]


def run_pipeline(config: dict, logger, steps: list):
    """
    按顺序执行指定的流水线步骤。

    参数:
        config: 配置字典
        logger: 日志记录器
        steps: 要执行的步骤列表
    """
    step_times = {}
    total_start = time.time()

    for step in steps:
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"步骤: {step}")
        logger.info("=" * 60)

        step_start = time.time()

        if step == "data":
            from data_loader_tushare import run_data_pipeline
            run_data_pipeline(config, logger)

        elif step == "features":
            from features import compute_all_factors
            compute_all_factors(config, logger)

        elif step == "labels":
            from labels import build_labels
            build_labels(config, logger)

        elif step == "preprocess":
            from preprocess import preprocess_factors
            preprocess_factors(config, logger)

        elif step == "split":
            from split import run_split
            run_split(config, logger)

        elif step == "models":
            from models import train_all_models
            train_all_models(config, logger)

        elif step == "backtest":
            from backtest import run_backtest
            run_backtest(config, logger)

        elif step == "evaluation":
            from evaluation import run_evaluation
            run_evaluation(config, logger)

        elif step == "analysis":
            from analysis import run_analysis
            run_analysis(config, logger)

        else:
            logger.warning(f"未知步骤: {step}，跳过")
            continue

        elapsed = time.time() - step_start
        step_times[step] = elapsed
        logger.info(f"步骤 {step} 完成，耗时 {elapsed:.1f}s")

    # 汇总
    total_elapsed = time.time() - total_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("流水线执行完毕")
    logger.info("=" * 60)
    logger.info(f"总耗时: {total_elapsed:.1f}s")
    for step, t in step_times.items():
        logger.info(f"  {step}: {t:.1f}s")
    logger.info(f"输出目录: {get_run_dir()}")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="多因子选股 + 树模型 统一流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
步骤列表:
  data         数据获取与清洗（tushare）
  features     因子计算
  labels       标签构建
  preprocess   因子预处理（去极值、标准化、中性化）
  split        时间序列滚动划分
  models       模型训练与预测
  backtest     回测与组合构建
  evaluation   评估指标（IC、ICIR、夏普等）
  analysis     归因分析（特征重要性、SHAP）

示例:
  python src/pipeline.py                              # 全流程
  python src/pipeline.py --skip data                  # 跳过数据下载
  python src/pipeline.py --only models,backtest,evaluation  # 只跑部分步骤
  python src/pipeline.py --run_id 3                   # 指定运行序号
        """
    )
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="配置文件路径（默认 config/config.yaml）")
    parser.add_argument("--run_id", type=int, default=None,
                        help="手动指定运行序号（默认自动递增）")
    parser.add_argument("--skip", type=str, default=None,
                        help="跳过指定步骤（逗号分隔，如 --skip data,analysis）")
    parser.add_argument("--only", type=str, default=None,
                        help="只运行指定步骤（逗号分隔，如 --only models,backtest")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 初始化运行
    run_id_str = init_run(config, args.run_id)

    # 初始化 logger
    logger = setup_logger("pipeline", config.get("log_level", "INFO"))
    logger.info(f"运行 ID: {run_id_str}")
    logger.info(f"输出目录: {get_run_dir()}")

    # 确定要执行的步骤
    if args.only:
        steps = [s.strip() for s in args.only.split(",")]
        # 验证步骤名
        for s in steps:
            if s not in ALL_STEPS:
                logger.error(f"未知步骤: {s}，可选: {', '.join(ALL_STEPS)}")
                return
    else:
        steps = list(ALL_STEPS)
        if args.skip:
            skip_set = {s.strip() for s in args.skip.split(",")}
            steps = [s for s in steps if s not in skip_set]

    logger.info(f"执行步骤: {' → '.join(steps)}")
    logger.info("")

    # 执行流水线
    run_pipeline(config, logger, steps)


if __name__ == "__main__":
    main()
