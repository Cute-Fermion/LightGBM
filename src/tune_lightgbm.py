#!/usr/bin/env python
"""
tune_lightgbm.py — LightGBM 贝叶斯超参数优化脚本
==================================================

使用 Optuna (TPE sampler) 对 LightGBM 进行超参数搜索。
优化目标：ICIR（IC 均值 / IC 标准差），同时记录 Rank IC 均值。

数据加载和预处理只做一次，每轮 trial 只重复"训练 + 评估"。

运行方式:
    python src/tune_lightgbm.py --config config/config.yaml --n_trials 50
    python src/tune_lightgbm.py --config config/config.yaml --n_trials 80 --seed 123
"""

import sys
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---- 项目路径设置 ----
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import load_config, setup_logger, get_data_path, init_run

warnings.filterwarnings("ignore")


# ============================================================
# LightGBM 超参数搜索空间
# ============================================================

def define_search_space(trial):
    """
    定义 Optuna 搜索空间。

    覆盖 LightGBM 的关键超参数，范围基于经验和学术文献：
    - 树复杂度：num_leaves, max_depth, min_child_samples
    - 学习率与迭代：learning_rate, n_estimators
    - 采样：subsample, colsample_bytree
    - 正则化：reg_lambda (L2), reg_alpha (L1)
    """
    return {
        # 树复杂度
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),

        # 学习率与迭代次数
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),

        # 采样比例
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),

        # 正则化
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
    }


# ============================================================
# 目标函数
# ============================================================

def make_objective(windows_data, config, logger):
    """
    构造 Optuna 目标函数。

    闭包捕获 windows_data（只加载一次），每轮 trial 只重复训练+评估。

    参数:
        windows_data: run_split 返回的窗口列表
        config: 配置字典
        logger: 日志记录器

    返回:
        目标函数，接受 Optuna trial，返回 ICIR
    """
    # 从 config 读取固定的随机种子
    random_state = config.get("random_seed", 42)
    early_stopping_rounds = config["models"]["lightgbm"].get("early_stopping_rounds", 50)

    def objective(trial):
        params = define_search_space(trial)
        ic_list = []

        for wdata in windows_data:
            X_train = wdata["X_train"]
            y_train = wdata["y_train"]
            X_valid = wdata["X_valid"]
            y_valid = wdata["y_valid"]

            # 跳过数据量不足的窗口
            if len(X_train) < 50 or len(X_valid) < 20:
                continue

            # 构建 LightGBM 模型（复用 models.py 中的 LightGBMModel）
            from models import LightGBMModel, calc_rank_ic

            model = LightGBMModel(
                params={**params, "random_state": random_state, "n_jobs": -1},
                early_stopping_rounds=early_stopping_rounds,
            )
            model.fit(X_train, y_train, X_valid, y_valid)
            y_pred = model.predict(X_valid)

            # 计算该窗口的 Rank IC
            ic = calc_rank_ic(y_valid, y_pred)
            if not np.isnan(ic):
                ic_list.append(ic)

        # 数据不足
        if len(ic_list) < 3:
            return -1.0

        ic_arr = np.array(ic_list)
        ic_mean = float(ic_arr.mean())
        ic_std = float(ic_arr.std())

        # ICIR = IC_mean / IC_std，加下限防止除零
        icir = ic_mean / max(ic_std, 0.01)

        # 在 user_attrs 中记录详细信息，方便后续分析
        trial.set_user_attr("ic_mean", ic_mean)
        trial.set_user_attr("ic_std", ic_std)
        trial.set_user_attr("n_windows", len(ic_list))

        return icir

    return objective


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LightGBM 贝叶斯超参数优化")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--n_trials", type=int, default=50, help="Optuna 试验次数（默认 50）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（默认用 config 中的）")
    parser.add_argument("--run_id", type=int, default=None, help="运行序号")
    args = parser.parse_args()

    config = load_config(args.config)
    run_id = init_run(config, args.run_id)
    logger = setup_logger("tune_lightgbm", config.get("log_level", "INFO"))

    logger.info("=" * 60)
    logger.info("LightGBM 贝叶斯超参数优化")
    logger.info(f"试验次数: {args.n_trials}")
    logger.info(f"运行 ID: {run_id}")
    logger.info("=" * 60)

    # ---- 检查预处理数据 ----
    proc_path = get_data_path(config, "processed")

    if not (proc_path / "factors_processed.parquet").exists():
        logger.info("预处理因子不存在，自动运行预处理...")
        from preprocess import preprocess_factors
        preprocess_factors(config, logger)

    if not (proc_path / "labels.parquet").exists():
        logger.error("标签文件不存在！请先运行标签构建: python src/labels.py --config config/config.yaml")
        sys.exit(1)

    # ---- 检查 Optuna ----
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)  # 关闭 Optuna 自身的详细日志
    except ImportError:
        logger.error("需要安装 Optuna: pip install optuna")
        sys.exit(1)

    # ---- 数据划分（只做一次）----
    from split import run_split

    logger.info("执行时间序列滚动划分（只做一次）...")
    windows_data = run_split(config, logger)
    logger.info(f"共 {len(windows_data)} 个滚动窗口")

    if not windows_data:
        logger.error("没有有效的滚动窗口！请检查 config 中的 split 配置")
        sys.exit(1)

    # ---- 运行 Optuna 优化 ----
    seed = args.seed if args.seed is not None else config.get("random_seed", 42)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="lightgbm_tune")

    objective = make_objective(windows_data, config, logger)

    logger.info(f"开始 Optuna 优化（{args.n_trials} 轮试验）...")
    logger.info("-" * 60)

    def _trial_log(study, trial):
        """每 10 轮输出一次进度"""
        if (trial.number + 1) % 10 == 0 or trial.number == 0:
            ic_mean = trial.user_attrs.get("ic_mean", float("nan"))
            logger.info(
                f"  Trial {trial.number + 1}/{args.n_trials}  "
                f"ICIR = {trial.value:.4f}  "
                f"IC = {ic_mean:.4f}  "
                f"best = {study.best_value:.4f}"
            )

    study.optimize(objective, n_trials=args.n_trials, callbacks=[_trial_log])

    # ---- 输出结果 ----
    logger.info("=" * 60)
    logger.info("优化完成！")
    logger.info(f"最优 ICIR: {study.best_value:.4f}")
    logger.info(f"最优 IC 均值: {study.best_trial.user_attrs.get('ic_mean', 0):.4f}")
    logger.info(f"最优参数:")
    for k, v in study.best_params.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    # ---- 保存结果 ----
    results_dir = get_data_path(config, "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. 完整优化结果 (JSON)
    result = {
        "best_icir": float(study.best_value),
        "best_ic_mean": float(study.best_trial.user_attrs.get("ic_mean", 0)),
        "best_ic_std": float(study.best_trial.user_attrs.get("ic_std", 0)),
        "best_params": study.best_params,
        "n_trials": args.n_trials,
        "n_windows": int(study.best_trial.user_attrs.get("n_windows", 0)),
        "seed": seed,
    }
    result_path = results_dir / "tune_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"优化结果已保存: {result_path}")

    # 2. 最优 LightGBM 参数 (YAML)，可直接替换 config 中的 lightgbm.params
    best_lgb_config = {
        "lightgbm": {
            "enabled": True,
            "params": {
                **{k: v for k, v in study.best_params.items() if k != "n_estimators"},
                "n_estimators": study.best_params.get("n_estimators", 500),
                "random_state": config.get("random_seed", 42),
                "n_jobs": -1,
            },
            "early_stopping_rounds": config["models"]["lightgbm"].get("early_stopping_rounds", 50),
        }
    }
    import yaml
    best_config_path = results_dir / "best_lightgbm_params.yaml"
    with open(best_config_path, "w", encoding="utf-8") as f:
        yaml.dump(best_lgb_config, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"最优参数已保存: {best_config_path}")

    # 3. 全部 trial 历史 (CSV)，方便事后分析
    trial_df = study.trials_dataframe(attrs=("value", "user_attrs"))
    history_path = results_dir / "tune_history.csv"
    trial_df.to_csv(history_path, index=False)
    logger.info(f"试验历史已保存: {history_path}")

    logger.info("全部完成！")


if __name__ == "__main__":
    main()
