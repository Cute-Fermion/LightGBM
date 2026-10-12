"""
models.py - 模型训练与预测模块
================================
实现四种模型的训练和预测：
    1. 线性回归（基线模型）
    2. 随机森林
    3. XGBoost
    4. LightGBM

每个模型在每个滚动窗口上独立训练和预测。
使用 RMSE 做早停（与树模型原生优化目标一致），
最终用 Rank IC 做模型选择和评估。

训练流程:
    对于每个滚动窗口:
    1. 用训练集训练模型
    2. 用验证集做早停/超参选择
    3. 对验证集生成预测值
    4. 保存模型和预测结果

运行方式:
    python src/models.py --config config/config.yaml
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from utils import (
    load_config, setup_logger, PROJECT_ROOT,
    get_data_path, get_output_path, ensure_dir, save_dataframe, load_dataframe
)
from split import run_split


# ============================================================
# 自定义评估指标：Rank IC（用于模型比较）
# ============================================================

def calc_rank_ic(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    计算 Rank IC：预测值与真实值的横截面 Spearman 相关系数。

    Rank IC 是选股模型最重要的指标：
    - 它衡量的是"排名预测"的准确性，而非绝对值
    - 值域 [-1, 1]，> 0.03 在 A 股就有选股能力
    - IC > 0.05 且 ICIR > 1.0 是较好的水平

    参数:
        y_true: 真实收益率
        y_pred: 预测值

    返回:
        Rank IC 值
    """
    valid = y_true.notna() & y_pred.notna()
    if valid.sum() < 10:
        return np.nan
    ic, _ = spearmanr(y_true[valid], y_pred[valid])
    return ic


# ============================================================
# 模型封装类
# ============================================================

class BaseModel:
    """
    模型基类：定义统一的训练/预测接口。

    所有模型（线性回归、随机森林、XGBoost、LightGBM）都继承此类。
    """

    def __init__(self, name: str, params: dict):
        self.name = name
        self.params = params
        self.model = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_valid: pd.DataFrame, y_valid: pd.Series):
        """训练模型"""
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """预测"""
        raise NotImplementedError

    def save(self, path: Path):
        """保存模型"""
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: Path):
        """加载模型"""
        with open(path, "rb") as f:
            self.model = pickle.load(f)


class LinearModel(BaseModel):
    """
    线性回归（Ridge 回归）。

    作为基线模型：
    - 如果树模型不能显著超越线性回归，说明非线性关系不强
    - Ridge 加了 L2 正则化，比 OLS 更稳定

    特点：
    - 不处理非线性关系
    - 训练速度快
    - 可解释性强（系数直接反映因子权重）
    """

    def __init__(self, params: dict = None):
        super().__init__("linear_regression", params or {})
        alpha = self.params.get("alpha", 1.0)
        self.model = Ridge(alpha=alpha, random_state=42)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        """线性回归不需要验证集做早停，直接全量训练"""
        self.model.fit(X_train.values, y_train.values)

    def predict(self, X):
        return pd.Series(self.model.predict(X.values), index=X.index)


class RandomForestModel(BaseModel):
    """
    随机森林回归。

    特点：
    - 天然处理非线性关系和交互效应
    - 不需要特征标准化
    - 但容易过拟合，需要控制树深度和叶子节点样本数
    - 训练速度比 GBDT 慢
    """

    def __init__(self, params: dict):
        super().__init__("random_forest", params)
        self.model = RandomForestRegressor(**params)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        self.model.fit(X_train.values, y_train.values)

    def predict(self, X):
        return pd.Series(self.model.predict(X.values), index=X.index)


class XGBoostModel(BaseModel):
    """
    XGBoost 模型。

    特点：
    - 梯度提升树，逐步减小残差
    - 支持自定义目标函数和评估指标
    - 内置早停机制，防止过拟合
    - 对缺失值有较好的处理能力

    参数说明:
    - n_estimators: 树的数量
    - max_depth: 单棵树最大深度（控制复杂度）
    - learning_rate: 学习率（越小越稳定，但需要更多树）
    - subsample: 每棵树使用的样本比例（随机性）
    - colsample_bytree: 每棵树使用的特征比例
    - reg_lambda: L2 正则化（防止过拟合）
    - reg_alpha: L1 正则化（特征选择）
    """

    def __init__(self, params: dict, early_stopping_rounds: int = 50):
        super().__init__("xgboost", params)
        self.early_stopping_rounds = early_stopping_rounds
        if xgb is None:
            raise ImportError("请安装 xgboost: pip install xgboost")
        self.model = xgb.XGBRegressor(
            objective="reg:squarederror",  # 回归任务，最小化 MSE
            **params
        )

    def fit(self, X_train, y_train, X_valid, y_valid):
        """
        训练 XGBoost，使用验证集做早停。
        早停指标：RMSE（与优化目标一致）
        """
        self.model.fit(
            X_train.values, y_train.values,
            eval_set=[(X_valid.values, y_valid.values)],
            verbose=False
        )

    def predict(self, X):
        return pd.Series(self.model.predict(X.values), index=X.index)


class LightGBMModel(BaseModel):
    """
    LightGBM 模型。

    特点：
    - 使用直方图算法，训练速度比 XGBoost 快
    - Leaf-wise 生长策略（比 level-wise 更高效）
    - 对大规模数据表现优秀
    - 是量化选股中最常用的树模型之一

    关键参数:
    - num_leaves: 每棵树的叶子节点数（控制复杂度，比 max_depth 更直接）
    - max_depth: 限制树深度（防止过拟合）
    - min_data_in_leaf: 叶子节点最小样本数
    """

    def __init__(self, params: dict, early_stopping_rounds: int = 50):
        super().__init__("lightgbm", params)
        self.early_stopping_rounds = early_stopping_rounds
        if lgb is None:
            raise ImportError("请安装 lightgbm: pip install lightgbm")
        self.model = lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            **params
        )

    def fit(self, X_train, y_train, X_valid, y_valid):
        """
        训练 LightGBM，使用验证集做早停。
        """
        self.model.fit(
            X_train.values, y_train.values,
            eval_set=[(X_valid.values, y_valid.values)],
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0)  # 不打印训练日志
            ]
        )

    def predict(self, X):
        return pd.Series(self.model.predict(X.values), index=X.index)


# ============================================================
# 模型训练主流程
# ============================================================

def train_all_models(config: dict, logger) -> Dict[str, pd.DataFrame]:
    """
    对所有模型、所有滚动窗口进行训练和预测。

    返回每个模型在所有验证窗口上的预测结果，用于后续评估。

    参数:
        config: 配置字典
        logger: 日志记录器

    返回:
        字典 {model_name: predictions_df}
        predictions_df 的 MultiIndex 为 (date, code)，列为 'pred' 和 'label'
    """
    # 获取滚动窗口的数据
    logger.info("生成滚动窗口数据...")
    windows_data = run_split(config, logger)

    if not windows_data:
        logger.error("没有生成任何滚动窗口！")
        return {}

    # 初始化模型
    models = {}
    model_configs = config["models"]

    # 线性回归
    if model_configs.get("linear_regression", {}).get("enabled", True):
        models["linear_regression"] = LinearModel()

    # 随机森林
    if model_configs.get("random_forest", {}).get("enabled", True):
        rf_params = model_configs["random_forest"]["params"]
        models["random_forest"] = RandomForestModel(rf_params)

    # XGBoost
    if model_configs.get("xgboost", {}).get("enabled", True):
        xgb_params = model_configs["xgboost"]["params"]
        esr = model_configs["xgboost"].get("early_stopping_rounds", 50)
        models["xgboost"] = XGBoostModel(xgb_params, esr)

    # LightGBM
    if model_configs.get("lightgbm", {}).get("enabled", True):
        lgb_params = model_configs["lightgbm"]["params"]
        esr = model_configs["lightgbm"].get("early_stopping_rounds", 50)
        models["lightgbm"] = LightGBMModel(lgb_params, esr)

    logger.info(f"将训练 {len(models)} 个模型: {list(models.keys())}")
    logger.info(f"共 {len(windows_data)} 个滚动窗口")

    # 保存目录
    model_save_dir = get_output_path(config, "results") / "models"
    pred_save_dir = get_output_path(config, "results") / "predictions"
    ensure_dir(model_save_dir)
    ensure_dir(pred_save_dir)

    # 对每个模型，在所有窗口上训练和预测
    all_predictions = {}

    for model_name, base_model in models.items():
        logger.info(f"\n{'='*40}")
        logger.info(f"训练模型: {model_name}")
        logger.info(f"{'='*40}")

        window_preds = []
        window_ics = []

        for wd in windows_data:
            window_id = wd["window_id"]
            X_train = wd["X_train"]
            y_train = wd["y_train"]
            X_valid = wd["X_valid"]
            y_valid = wd["y_valid"]

            if len(X_train) == 0 or len(X_valid) == 0:
                logger.warning(f"  窗口 {window_id}: 训练或验证集为空，跳过")
                continue

            # 检查是否已有保存的模型（断点续训）
            model_path = model_save_dir / f"{model_name}_window{window_id}.pkl"
            if model_path.exists():
                logger.info(f"  窗口 {window_id}: 检测到已保存的模型，加载并跳过训练")
                try:
                    if model_name == "linear_regression":
                        model = LinearModel()
                    elif model_name == "random_forest":
                        model = RandomForestModel(model_configs["random_forest"]["params"])
                    elif model_name == "xgboost":
                        model = XGBoostModel(
                            model_configs["xgboost"]["params"],
                            model_configs["xgboost"].get("early_stopping_rounds", 50)
                        )
                    elif model_name == "lightgbm":
                        model = LightGBMModel(
                            model_configs["lightgbm"]["params"],
                            model_configs["lightgbm"].get("early_stopping_rounds", 50)
                        )
                    model.load(model_path)
                    y_pred = model.predict(X_valid)
                    ic = calc_rank_ic(y_valid, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
                    window_ics.append(ic)
                    logger.info(f"  窗口 {window_id} (已加载): IC={ic:.4f}, RMSE={rmse:.4f}")
                    pred_df = pd.DataFrame({"pred": y_pred, "label": y_valid})
                    pred_df["window_id"] = window_id
                    window_preds.append(pred_df)
                    continue
                except Exception as e:
                    logger.warning(f"  窗口 {window_id}: 加载模型失败 ({e})，重新训练")

            # 每个窗口重新初始化模型（避免模型状态残留）
            if model_name == "linear_regression":
                model = LinearModel()
            elif model_name == "random_forest":
                model = RandomForestModel(model_configs["random_forest"]["params"])
            elif model_name == "xgboost":
                model = XGBoostModel(
                    model_configs["xgboost"]["params"],
                    model_configs["xgboost"].get("early_stopping_rounds", 50)
                )
            elif model_name == "lightgbm":
                model = LightGBMModel(
                    model_configs["lightgbm"]["params"],
                    model_configs["lightgbm"].get("early_stopping_rounds", 50)
                )

            # 训练
            try:
                model.fit(X_train, y_train, X_valid, y_valid)
            except Exception as e:
                logger.error(f"  窗口 {window_id} 训练失败: {e}")
                continue

            # 预测
            y_pred = model.predict(X_valid)

            # 计算 Rank IC
            ic = calc_rank_ic(y_valid, y_pred)
            rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
            window_ics.append(ic)

            logger.info(f"  窗口 {window_id}: IC={ic:.4f}, RMSE={rmse:.4f}, "
                         f"训练={len(X_train)}, 验证={len(X_valid)}")

            # 保存预测结果
            pred_df = pd.DataFrame({
                "pred": y_pred,
                "label": y_valid
            })
            pred_df["window_id"] = window_id
            window_preds.append(pred_df)

            # 保存模型（最后一个窗口的模型）
            model.save(model_save_dir / f"{model_name}_window{window_id}.pkl")

        # 合并所有窗口的预测结果
        if window_preds:
            predictions = pd.concat(window_preds)
            all_predictions[model_name] = predictions

            # 汇总统计
            mean_ic = np.nanmean(window_ics)
            std_ic = np.nanstd(window_ics)
            icir = mean_ic / std_ic if std_ic > 0 else 0

            logger.info(f"\n  {model_name} 汇总: "
                         f"平均IC={mean_ic:.4f}, IC标准差={std_ic:.4f}, ICIR={icir:.4f}")

            # 保存预测结果
            save_dataframe(predictions, pred_save_dir / f"{model_name}_predictions.parquet", logger)

    return all_predictions


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="模型训练与预测")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路径")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logger("models", config.get("log_level", "INFO"))

    logger.info("=" * 60)
    logger.info("模型训练开始")
    logger.info("=" * 60)

    predictions = train_all_models(config, logger)

    logger.info("\n" + "=" * 60)
    logger.info("模型训练完成！")
    for name, pred in predictions.items():
        logger.info(f"  {name}: {len(pred)} 条预测记录")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
