
# 多因子选股 + 树模型 项目开发需求说明

## 一、项目目标

在 A 股市场上，用 XGBoost / LightGBM 等树模型，在每个横截面上预测股票未来 20 日（月度）的相对收益排名，构建多空组合并进行回测，评估选股能力。

最终输出：
- 可复现的完整代码仓库
- 模型对比结果（线性回归 / 随机森林 / XGBoost / LightGBM）
- 回测报告（Rank IC、ICIR、夏普、最大回撤、换手率）
- 因子归因分析（特征重要性 + SHAP）

---

## 二、技术栈与依赖

- Python 3.9+
- 数据获取：`tushare`（优先，需 pro 账号）、`akshare`（备用/回退，免费无需 token）
- 数据处理：`pandas`、`numpy`
- 模型：`scikit-learn`、`xgboost`、`lightgbm`
- 评估与回测：`scipy`、`statsmodels`、`matplotlib`、`seaborn`
- 模型解释：`shap`
- 其他：`tqdm`、`joblib`、`pyyaml`

全部模块自行实现，不依赖 qlib / backtrader 等框架，以保证对每个环节的完全控制和可理解性。

---

## 三、项目目录结构

```
project/
├── config/
│   └── config.yaml              # 数据路径、模型参数、回测参数
├── data/
│   ├── raw/                     # 原始数据
│   └── processed/               # 处理后数据
├── src/
│   ├── utils.py                 # 工具函数（配置加载、日志、路径管理）
│   ├── data_loader.py           # 数据获取与清洗（akshare 版本）
│   ├── data_loader_tushare.py   # 数据获取与清洗（tushare 优先版本）
│   ├── features.py              # 因子计算
│   ├── labels.py                # 标签构建
│   ├── preprocess.py            # 去极值、标准化、中性化
│   ├── split.py                 # 时间序列滚动划分
│   ├── models.py                # 模型训练与预测
│   ├── backtest.py              # 回测与组合构建
│   ├── evaluation.py            # Rank IC、ICIR、夏普等
│   └── analysis.py              # 特征重要性、SHAP
├── notebooks/
│   └── demo.ipynb               # 完整流程演示
├── outputs/
│   ├── figures/                 # 图表
│   └── results/                 # 回测结果
├── requirements.txt
└── README.md
```

---

## 四、数据获取要求

### 数据源
- 优先使用 `tushare` pro 接口（需注册账号并配置 token），对于 tushare 无法覆盖的数据自动回退到 `akshare`。
- 两个数据入口保持完全相同的函数签名和输出格式，下游模块无需区分数据来源：
  - `data_loader.py`：纯 akshare 版本（保留备用）
  - `data_loader_tushare.py`：tushare 优先版本（当前使用）
- Tushare 关键差异：
  - `daily` / `daily_basic` / `index_daily` 等接口按交易日批量获取（一次调用获取全市场），比逐只下载快数十倍
  - 前复权价格通过 `adj_factor` 手动计算（不复权价格 × 复权因子）
  - 估值指标（PE/PB/市值等）来自 `daily_basic`，无需逐只请求
  - 行业分类回退到 akshare（tushare 的申万分类接口需要更高积分）
  - Tushare 用 `vol` 表示成交量，代码中统一重命名为 `volume`
- 历史成分股：使用当期成分股近似，接受约 1-3% 年化偏差。
- 财务数据发布日：使用固定延迟假设（年报 120 天、半年报 60 天、季报 30 天）近似。
- 数据拉取后存为本地 parquet 文件，后续流程直接读本地，不重复请求网络。
- Tushare 配置在 `config.yaml` 的 `tushare` 段：token、频率限制、重试参数。

### 数据范围
- 股票池：沪深 300 或中证 500 成分股，使用**历史成分股**，避免幸存者偏差。
- 时间范围：2018-01-01 至 2023-12-31。
- 频率：日频。

### 需要的数据字段
- 行情：开盘、收盘、最高、最低、成交量、成交额、换手率、复权因子。
- 财务：PE、PB、PS、ROE、ROA、毛利率、净利率、负债率、流动比率、营收增速。
- 状态：停牌标记、涨跌停标记。
- 指数：沪深 300 或中证 500 指数收盘价，用于计算超额收益。
- 行业：申万一级行业分类，用于中性化。

### 数据清洗
- 剔除上市不足 60 日的股票。
- 剔除停牌日。
- 剔除当日涨跌停的样本（回测时）。
- 财务数据按**发布日**对齐，避免未来信息泄露。

---

## 五、因子工程要求

### 因子类别与具体因子

**动量类**
- 过去 5/10/20/60 日收益率
- 过去 20 日收益率标准差
- 52 周高点距离 = 当前价 / 过去 252 日最高价 - 1

**反转类**
- 过去 1 日收益率
- 过去 5 日收益率（A 股短期反转效应显著）

**波动类**
- 过去 20 日已实现波动率
- ATR（14 日）
- 下行波动率（过去 20 日负收益标准差）

**量价类**
- 换手率
- 成交量变化率（5 日 / 20 日）
- 量价相关性（过去 20 日收益率与成交量相关系数）

**估值类**
- PE 倒数、PB 倒数、PS 倒数
- 股息率
- 市值对数

**质量类**
- ROE、ROA
- 毛利率、净利率
- 负债率、流动比率

### 因子处理流程
1. **去极值**：MAD 法，中位数 ± 3 倍 MAD。
2. **标准化**：横截面 z-score 或 rank 标准化。
3. **中性化**：对市值和行业做线性回归，取残差。
4. **缺失值**：保留，树模型可自动处理；缺失比例 > 50% 的因子剔除。

### 关键要求
- 所有因子计算**只用 t 时刻及之前的数据**。
- 因子计算函数需支持**增量计算**，避免全量重算。
- 每个因子需有**单元测试**，验证无未来信息泄露。

---

## 六、标签构建要求

- 预测目标：未来 20 日收益率（与月度调仓频率对齐）。
- 横截面标准化：减去当日全市场均值，再除以标准差。
- 标签计算：
  ```
  forward_return = (close.shift(-N) / close - 1)
  label = forward_return - forward_return.mean(axis=1)
  label = label / label.std(axis=1)
  ```
- 避免标签重叠：训练样本之间用 purge / embargo 间隔。

---

## 七、时间序列交叉验证要求

### 划分方式
- **滚动窗口**：用过去 3 年训练，预测下 1 个月，窗口每月向前滚动。
- **扩展窗口**：用过去所有数据训练，预测下 1 个月。
- **Purged K-Fold**：在 K 折基础上，剔除与验证集时间重叠的训练样本。

### 推荐
- 默认使用滚动窗口 + 每月调仓。
- 在 `config.yaml` 中可配置训练窗口长度、预测窗口长度、滚动步长。

---

## 八、模型训练要求

### 对比模型
1. 线性回归（基线）
2. 随机森林
3. XGBoost
4. LightGBM

### 关键参数（在 config.yaml 中可配置）

**XGBoost**
```yaml
xgboost:
  n_estimators: 500
  max_depth: 5
  learning_rate: 0.05
  subsample: 0.8
  colsample_bytree: 0.8
  reg_lambda: 1.0
  reg_alpha: 0.0
  early_stopping_rounds: 50
```

**LightGBM**
```yaml
lightgbm:
  n_estimators: 500
  num_leaves: 31
  max_depth: 5
  learning_rate: 0.05
  subsample: 0.8
  colsample_bytree: 0.8
  reg_lambda: 1.0
  reg_alpha: 0.0
  early_stopping_rounds: 50
```

### 训练要求
- 早停指标使用 **RMSE**（与树模型原生优化目标一致，训练更稳定）；**Rank IC** 用于模型选择和最终评估。
- 每期重新训练，或每季度重新训练。
- 特征做横截面标准化后再喂给模型。
- 保存每期的模型和预测结果。

---

## 九、回测要求

### 组合构建
- 每期按预测值从高到低分 5 组。
- 做多 top 组，做空 bottom 组（A 股做空难，可只做多 top 组）。
- 等权或市值加权，在 config.yaml 中可配置。
- 每月调仓。

### 回测细节
- **交易成本**：佣金（买卖各 0.3‰）+ 印花税（卖出 0.5‰）+ 滑点（0.1%）。
- **涨跌停**：涨停不能买，跌停不能卖。
- **停牌**：停牌期间不交易，复牌后按开盘价成交。
- **幸存者偏差**：用历史成分股，包含已退市股票。
- **滑点**：可配置，默认 0.1%。

### 输出指标
- 年化收益、年化波动、夏普比率
- 最大回撤、Calmar 比率
- 换手率、交易成本占比
- 分年度收益
- 多空组合净值曲线

---

## 十、评估指标要求

### 核心指标
- **Rank IC**：预测值与未来收益的横截面 Spearman 相关系数。
- **ICIR**：IC 均值 / IC 标准差。
- **IC 衰减**：不同持有期的 IC 变化。

### 辅助指标
- RMSE、MAE（回归任务）
- 分组单调性（5 组收益是否单调）
- 多空收益差

---

## 十一、归因分析要求

- **特征重要性**：XGBoost / LightGBM 的 `feature_importances_`。
- **SHAP 值**：解释单个样本的预测，看因子是非线性还是线性贡献。
- **因子相关性**：看模型是否过度依赖某几个因子。
- **与经典因子对照**：如果动量因子贡献最大，与学术文献一致，说明模型合理。

---

## 十二、代码质量要求

- 所有函数需有 docstring。
- 关键步骤需有日志输出。
- 配置文件与代码分离。
- 每个模块需有单元测试。
- 代码需支持命令行运行，如：
  ```
  python src/models.py --config config/config.yaml
  ```
- 结果需保存到 outputs/ 目录。

---

## 十三、README 要求

README 需包含：
- 项目简介
- 数据来源与时间范围
- 因子列表与构建逻辑
- 防泄露设计说明
- 模型对比结果
- 回测表现
- 归因结论
- 复现步骤

---

## 十四、开发顺序建议

1. 工具函数（utils.py）
2. 数据获取与清洗（data_loader_tushare.py 优先，data_loader.py 备用）
3. 因子计算（features.py）
4. 标签构建（labels.py）
5. 预处理（preprocess.py）
6. 时间序列划分（split.py）
7. 模型训练（models.py）
8. 回测（backtest.py）
9. 评估（evaluation.py）
10. 归因分析（analysis.py）
11. 整合到 demo.ipynb

---

## 十五、注意事项

- 所有涉及未来数据的操作必须严格禁止。
- 财务数据必须按发布日对齐。
- 回测必须考虑交易成本、涨跌停、停牌。
- 模型评估必须用 Rank IC，而不是 RMSE 或 AUC。
- 代码需可复现，随机种子固定。
