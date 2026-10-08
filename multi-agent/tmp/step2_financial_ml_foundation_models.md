# Step 2 — 金融 ML 基础模型行业实践检索

AS_OF: 2026-10-08 23:16（检索时间；以下为公开研究/论文/行业报告，非实时行情）

---

## 1. 金融时间序列基础模型（TSFM）全景

### 1.1 通用型 TSFM（非金融专用）

| 模型 | 开发者 | 发布时间 | 架构 | 参数量 | 金融表现 |
|------|--------|----------|------|--------|----------|
| **TimesFM** | Google Research | 2024-01（2.5: 2025-09, 3.0: 2026-08） | Transformer (Decoder-only) | 500M | **R² = -2.80%**（金融零样本预测，负值=比均值预测还差） |
| **Chronos** | Amazon Science | 2024-03 | Transformer (T5-based) | 120M-710M | **R² = -1.37%**（金融零样本预测，同样无效） |
| **Moirai** | Salesforce | 2024-10（2.0: 2025-11） | Transformer (Anomaly Detection) | 100M-1B | 未专门测试金融数据 |
| **TimeGPT** | Nixtla | 2023-11 | Transformer-based | 未公开 | 支持 exogenous variables，但金融表现未公开 |
| **Lag-LLama** | ServiceNow | 2024-02 | Transformer (Patch-based) | 100M-1B | 未专门测试金融数据 |
| **Timer-XL** | THUML (清华) | 2024-06 | Transformer (Cross-attention) | 1B | 未专门测试金融数据 |

### 1.2 关键发现：通用 TSFM 在金融数据上表现极差

**来源**：arXiv:2511.18578 "Re(Visiting) Time Series Foundation Models in Finance"（2025）

> "The underperformance of general-purpose TSFMs on financial data is no accident. TimesFM (500M parameters) achieved a financial zero-shot forecast R² of **-2.80%** and a return of just **-1.47%**. Chronos (large) fared no better at **R² = -1.37%**, an effectively meaningless result. These models are not bad — they simply were never designed for financial data."

**含义**：
- 通用 TSFM（TimesFM、Chronos 等）在金融数据上**零样本预测无效**（R² 为负）
- 原因：金融数据具有**高噪声、对抗性、频繁 regime shift**，与零售/制造/医疗等场景差异巨大
- **XTX 的"foundation model"必须是金融专用**，而非直接套用通用 TSFM

---

## 2. 金融专用 TSFM（Domain-Specific）

### 2.1 Kronos（金融领域基础模型）

**来源**：arXiv:2508.02739 "Kronos: A Foundation Model for Time Series in Financial Domain"（AAAI 2026）

| 维度 | 详情 |
|------|------|
| **架构** | Transformer（Decoder-only） |
| **输入** | K-line（OHLCV）作为 token |
| **输出** | P(K_{t+1:K} \| K_{1:t})——给定历史 K 线，预测未来 K 线的概率分布 |
| **训练数据** | 金融 K 线数据（具体来源未公开） |
| **优势** | 金融专用，零样本预测优于通用 TSFM |
| **局限** | 仅支持 K 线数据，不支持 exogenous variables |

### 2.2 FinCast（金融预测基础模型）

**来源**：Pebblous 博客（2026）

| 维度 | 详情 |
|------|------|
| **架构** | Transformer-based |
| **输入** | 金融时间序列 + 宏观指标 |
| **输出** | 价格预测 + 置信区间 |
| **优势** | 支持多变量（multivariate），可融合宏观数据 |
| **局限** | 公开信息有限 |

### 2.3 关键结论

> "The TSFM ecosystem underwent rapid divergence during 2024-2025. On one front, tech giants like Google (TimesFM), Amazon (Chronos), and Salesforce (Moirai) are competitively advancing general-purpose TSFMs. On another, **domain-specific models like Kronos and FinCast are opening new fronts by demonstrating performance that surpasses general-purpose models in targeted domains**."

**含义**：
- 2024-2025 年 TSFM 生态分化为两条路线：
  1. **通用型**（Google/Amazon/Salesforce）：追求跨领域泛化
  2. **领域专用型**（Kronos/FinCast）：追求特定领域（金融）精度
- **XTX 的"foundation model"属于后者**——金融专用，而非通用

---

## 3. LLM + Transformer 混合架构（学术前沿）

### 3.1 LLM-Transformer 混合架构

**来源**：ICCK Transactions on Intelligent Systematics（2026-04）

> "This study proposes a novel **synergistic LLM-Transformer architecture** for stock price prediction. The LLM extracts sentiment from financial news, while the Transformer models price dynamics. The hybrid approach outperforms standalone models."

| 组件 | 作用 |
|------|------|
| **LLM** | 从新闻/社交媒体提取情绪信号 |
| **Transformer** | 建模价格动态（时间序列） |
| **融合** | 情绪信号 + 价格信号 → 预测 |

### 3.2 LLM 生成 Formulaic Alpha

**来源**：Digital Finance（Springer, 2026-06）

> "Traditionally, traders and quantitative analysts address alpha decay by manually crafting formulaic alphas... With recent advances in LLMs, it is now possible to **automate the generation of such alphas** by leveraging the reasoning capabilities of LLMs."

| 维度 | 详情 |
|------|------|
| **输入** | 历史数据 + 市场状态 |
| **LLM 作用** | 自动生成数学表达式（alpha 因子） |
| **输出** | 可执行的交易信号 |
| **优势** | 自动化、可扩展、适应 regime shift |

### 3.3 LLM-Augmented Linear Transformer-CNN

**来源**：MDPI Mathematics（2025-01）

> "We propose a novel hybrid deep learning framework that integrates a **large language model (LLM), a Linear Transformer (LT), and a Convolutional Neural Network (CNN)** to enhance stock price prediction using solely historical market data. The framework leverages the LLM as a professional financial analyst to perform daily technical analysis."

| 组件 | 作用 |
|------|------|
| **LLM** | 作为"专业金融分析师"，执行每日技术分析 |
| **Linear Transformer** | 建模长期依赖 |
| **CNN** | 提取局部特征 |
| **输出** | 价格预测 |

### 3.4 对冲基金视角的 LLM 综述

**来源**：arXiv:2605.05211 "A Review of Large Language Models for Stock Price Forecasting from a Hedge-Fund Perspective"（2026）

> "Large language models (LLMs) are increasingly deployed in quantitative finance for stock price forecasting. This review synthesizes recent applications of LLMs in this domain, including **extracting sentiment from financial news and social media, analyzing financial reports and earnings-call transcripts, tokenizing or symbolizing stock price series, and constructing multi-agent trading systems**."

**关键发现**：
- LLM 在量化金融中的应用包括：
  1. **情绪提取**（新闻/社交媒体）
  2. **财报/电话会议分析**
  3. **价格序列 tokenization**
  4. **多智能体交易系统**
- **实际陷阱**：
  - 流动性溢价（illiquidity premium）被低估
  - 回测过拟合
  - 实盘表现与回测差异巨大

---

## 4. XTX 方法的行业定位

### 4.1 XTX vs. 通用 TSFM

| 维度 | 通用 TSFM（TimesFM/Chronos） | XTX Foundation Model |
|------|-------------------------------|----------------------|
| **设计目标** | 跨领域泛化 | **金融专用** |
| **训练数据** | 零售/制造/医疗/金融混合 | **53,000+ 金融工具** |
| **金融表现** | R² = -2.80%（无效） | **未公开，但日交易量 >$300B** |
| **算力** | 单 GPU 推理 | **25,000+ GPU 训练** |
| **执行** | 离线预测 | **全自动实时交易** |

### 4.2 XTX vs. 金融专用 TSFM（Kronos/FinCast）

| 维度 | Kronos/FinCast | XTX Foundation Model |
|------|----------------|----------------------|
| **公开程度** | 论文/代码公开 | **完全保密** |
| **数据规模** | 未公开 | **650 PB** |
| **算力** | 学术级 | **25,000+ GPU** |
| **执行** | 研究/演示 | **实盘交易 >$300B/日** |
| **验证** | 回测 | **实盘 P&L** |

### 4.3 XTX 的独特性

1. **规模**：25,000+ GPU、650 PB 存储——远超学术/创业公司
2. **实盘验证**：日交易量 >$300B，非回测
3. **全自动**：无人工干预，纯模型驱动
4. **保密**：无公开论文/代码，技术细节完全保密
5. **金融专用**：非通用 TSFM，而是为金融数据专门设计

---

## 5. 主要来源

1. Google Research TimesFM GitHub: https://github.com/google-research/timesfm
2. Google Research Blog (TimesFM-3): https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting
3. Pebblous (Kronos 分析): https://blog.pebblous.ai/report/kronos-financial-foundation-model/en
4. arXiv:2511.18578 (Re(Visiting) TSFM in Finance): https://arxiv.org/abs/2511.18578
5. arXiv:2508.02739 (Kronos): https://arxiv.org/abs/2508.02739
6. ICCK (LLM-Transformer 混合): https://www.icck.org/article/abs/tis.2025.976754
7. Digital Finance (LLM Alpha): https://ideas.repec.org/a/spr/digfin/v8y2026i2d10.1007_s42521-026-00176-5.html
8. MDPI (LLM-Augmented LT-CNN): https://www.mdpi.com/2227-7390/13/3/487
9. arXiv:2605.05211 (LLM 对冲基金综述): https://arxiv.org/html/2605.05211v1
10. Medium (TSFM 综述): https://mychen76.medium.com/time-series-foundation-models-for-forecasting-task-c9076cae9a84
