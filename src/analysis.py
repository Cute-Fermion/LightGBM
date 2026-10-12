"""
analysis.py - 归因分析模块
==============================
对训练好的模型进行归因分析，理解模型"为什么"做出这样的预测。

分析内容:
    1. 特征重要性（Feature Importance）
       - 树模型内置的特征重要性（基于信息增益）
       - 哪些因子对预测贡献最大？

    2. SHAP 值分析
       - 每个因子对每个样本预测的贡献方向和大小
       - 因子是非线性贡献还是线性贡献？
       - 因子之间有没有交互效应？

    3. 因子相关性分析
       - 因子之间是否高度相关？
       - 模型是否过度依赖某几个因子？

    4. 与经典因子对照
       - 如果动量因子贡献最大，与学术文献一致，说明模型合理

运行方式:
    python src/analysis.py --config config/config.yaml
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，服务器环境也能画图
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional

try:
    import shap
except ImportError:
    shap = None

from utils import (
    load_config, setup_logger, PROJECT_ROOT,
    get_data_path, get_output_path, ensure_dir, load_dataframe
)


# ============================================================
# 第一部分：特征重要性
# ============================================================

def extract_feature_importance(
    model,
    feature_names: list,
    model_name: str,
    logger
) -> pd.Series:
    """
    提取模型的特征重要性。

    对于树模型（XGBoost/LightGBM/随机森林），使用内置的 feature_importances_。
    对于线性模型，使用系数的绝对值。

    参数:
        model: 训练好的模型对象
        feature_names: 特征名称列表
        model_name: 模型名称
        logger: 日志记录器

    返回:
        特征重要性 Series，index 为特征名
    """
    try:
        if hasattr(model, "feature_importances_"):
            # 树模型：使用内置的特征重要性
            importance = model.feature_importances_
            logger.info(f"  {model_name}: 使用 feature_importances_")
        elif hasattr(model, "coef_"):
            # 线性模型：使用系数绝对值
            importance = np.abs(model.coef_)
            logger.info(f"  {model_name}: 使用 |coef_|")
        else:
            logger.warning(f"  {model_name}: 无法提取特征重要性")
            return pd.Series(dtype=float)

        result = pd.Series(importance, index=feature_names)
        result = result.sort_values(ascending=False)

        return result

    except Exception as e:
        logger.error(f"  提取特征重要性失败: {e}")
        return pd.Series(dtype=float)


def plot_feature_importance(
    importance: pd.Series,
    model_name: str,
    output_dir: Path,
    top_n: int = 20,
    logger=None
):
    """
    绘制特征重要性柱状图。

    参数:
        importance: 特征重要性 Series
        model_name: 模型名称
        output_dir: 图表输出目录
        top_n: 显示前 N 个特征
        logger: 日志记录器
    """
    if len(importance) == 0:
        return

    # 取 Top N
    top_imp = importance.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    top_imp.sort_values().plot(kind="barh", ax=ax)
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"{model_name} - Top {top_n} Feature Importance")
    plt.tight_layout()

    save_path = output_dir / f"{model_name}_feature_importance.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if logger:
        logger.info(f"  特征重要性图已保存: {save_path}")


# ============================================================
# 第二部分：SHAP 分析
# ============================================================

def run_shap_analysis(
    model,
    X_data: pd.DataFrame,
    model_name: str,
    output_dir: Path,
    max_samples: int = 1000,
    logger=None
) -> Optional[pd.DataFrame]:
    """
    运行 SHAP 分析。

    SHAP（SHapley Additive exPlanations）基于博弈论：
    - 每个特征的 SHAP 值表示该特征对预测结果的边际贡献
    - SHAP 值之和 = 模型预测值 - 基准值（所有样本的均值）
    - 正 SHAP 值表示该特征推高了预测值
    - 负 SHAP 值表示该特征拉低了预测值

    参数:
        model: 训练好的模型
        X_data: 特征数据（用于计算 SHAP 值）
        model_name: 模型名称
        output_dir: 输出目录
        max_samples: 最多计算多少个样本的 SHAP 值（SHAP 计算很慢）
        logger: 日志记录器

    返回:
        SHAP 值 DataFrame（样本×特征），或 None
    """
    if shap is None:
        if logger:
            logger.warning("  shap 未安装，跳过 SHAP 分析")
        return None

    if len(X_data) == 0:
        return None

    # 限制样本数量（SHAP 计算很慢）
    if len(X_data) > max_samples:
        X_sample = X_data.sample(n=max_samples, random_state=42)
    else:
        X_sample = X_data

    if logger:
        logger.info(f"  计算 SHAP 值（{len(X_sample)} 个样本）...")

    try:
        # 根据模型类型选择 SHAP 解释器
        if model_name in ["xgboost", "lightgbm"]:
            # 树模型使用 TreeExplainer（精确且快速）
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        elif model_name == "random_forest":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            # 随机森林的 SHAP 可能返回多个数组（多棵树），取平均
            if isinstance(shap_values, list):
                shap_values = np.mean(shap_values, axis=0)
        else:
            # 线性模型使用 LinearExplainer 或 KernelExplainer
            # KernelExplainer 很慢，这里跳过
            if logger:
                logger.info(f"  {model_name} 跳过 SHAP 分析（线性模型计算量大）")
            return None

        # 转为 DataFrame
        shap_df = pd.DataFrame(shap_values, columns=X_sample.columns, index=X_sample.index)

        # ----- 绘制 SHAP 摘要图 -----
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
        plt.title(f"{model_name} - SHAP Summary")
        plt.tight_layout()

        save_path = output_dir / f"{model_name}_shap_summary.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        if logger:
            logger.info(f"  SHAP 摘要图已保存: {save_path}")

        # ----- 绘制 SHAP 条形图（平均绝对 SHAP） -----
        mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 8))
        mean_abs_shap.head(20).sort_values().plot(kind="barh", ax=ax)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"{model_name} - Mean Absolute SHAP")
        plt.tight_layout()

        save_path = output_dir / f"{model_name}_shap_bar.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        if logger:
            logger.info(f"  SHAP 条形图已保存: {save_path}")

        return shap_df

    except Exception as e:
        if logger:
            logger.error(f"  SHAP 分析失败: {e}")
        return None


# ============================================================
# 第三部分：因子相关性分析
# ============================================================

def analyze_factor_correlation(
    factors: pd.DataFrame,
    output_dir: Path,
    logger=None
):
    """
    分析因子之间的相关性。

    目的：
    - 如果两个因子高度相关（如 > 0.8），说明它们提供了冗余信息
    - 可以考虑剔除其中一个，减少多重共线性
    - 观察模型是否过度依赖某几个高度相关的因子

    参数:
        factors: 因子 DataFrame
        output_dir: 输出目录
        logger: 日志记录器
    """
    # 计算相关系数矩阵
    corr_matrix = factors.corr()

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        ax=ax,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Factor Correlation Matrix")
    plt.tight_layout()

    save_path = output_dir / "factor_correlation.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if logger:
        logger.info(f"  因子相关性热力图已保存: {save_path}")

    # 找出高度相关的因子对
    high_corr_pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append({
                    "factor_1": cols[i],
                    "factor_2": cols[j],
                    "correlation": corr_val
                })

    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs).sort_values("correlation", ascending=False)
        if logger:
            logger.info(f"  高度相关的因子对（|corr| > 0.7）:")
            for _, row in high_corr_df.iterrows():
                logger.info(f"    {row['factor_1']} <-> {row['factor_2']}: {row['correlation']:.3f}")

        # 保存
        high_corr_df.to_csv(output_dir / "high_correlation_pairs.csv", index=False)

    return corr_matrix


# ============================================================
# 第四部分：主流程
# ============================================================

def run_analysis(config: dict, logger):
    """
    执行完整的归因分析流程。

    参数:
        config: 配置字典
        logger: 日志记录器
    """
    import pickle

    proc_path = get_data_path(config, "processed")
    model_dir = get_output_path(config, "results") / "models"
    output_dir = get_output_path(config, "figures")
    ensure_dir(output_dir)

    # 加载因子数据（用于相关性分析）
    logger.info("加载因子数据...")
    factors = load_dataframe(proc_path / "factors_processed.parquet", logger)

    # 因子相关性分析（全局）
    logger.info("\n分析因子相关性...")
    analyze_factor_correlation(factors, output_dir, logger)

    # 对每个模型做特征重要性和 SHAP 分析
    model_files = list(model_dir.glob("*.pkl"))
    if not model_files:
        logger.warning("未找到模型文件，跳过特征重要性和 SHAP 分析")
        return

    # 按模型名分组，每个模型取最后一个窗口的模型
    model_groups = {}
    for f in model_files:
        # 文件名格式: {model_name}_window{N}.pkl
        name = f.stem
        parts = name.rsplit("_window", 1)
        if len(parts) == 2:
            model_name = parts[0]
            window_id = int(parts[1])
            if model_name not in model_groups or window_id > model_groups[model_name][1]:
                model_groups[model_name] = (f, window_id)

    for model_name, (model_file, window_id) in model_groups.items():
        logger.info(f"\n分析模型: {model_name} (窗口 {window_id})")

        # 加载模型
        with open(model_file, "rb") as f:
            model = pickle.load(f)

        # 获取特征名
        feature_names = factors.columns.tolist()

        # 提取特征重要性
        importance = extract_feature_importance(model, feature_names, model_name, logger)
        if len(importance) > 0:
            plot_feature_importance(importance, model_name, output_dir, logger=logger)

            # 保存特征重要性
            importance.to_csv(output_dir / f"{model_name}_feature_importance.csv")
            logger.info(f"  Top 5 特征:")
            for feat, val in importance.head(5).items():
                logger.info(f"    {feat}: {val:.4f}")

        # SHAP 分析（使用最后一个窗口的验证集数据）
        # 为了简化，使用因子数据的一个子集
        X_sample = factors.dropna().tail(500)
        if len(X_sample) > 0:
            run_shap_analysis(model, X_sample, model_name, output_dir, logger=logger)

    logger.info(f"\n归因分析完成！图表保存在: {output_dir}")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="归因分析")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路径")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logger("analysis", config.get("log_level", "INFO"))

    logger.info("=" * 60)
    logger.info("归因分析开始")
    logger.info("=" * 60)

    run_analysis(config, logger)

    logger.info("\n" + "=" * 60)
    logger.info("归因分析完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
