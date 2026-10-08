# Step 1 — Atlas Wang 演讲与 XTX 技术细节检索

AS_OF: 2026-10-08 23:14（检索时间；以下为公开报道/学术/官网信息，非实时行情）

---

## 1. Atlas Wang 演讲（Stony Brook, 2026-03-09）

### 完整摘要（Stony Brook AI3 官网）
> "At XTX Markets, we view algorithmic trading as one of the most compelling real-world frontiers for **deep learning and foundation models**. Every day, our systems generate forecasts for tens of thousands of financial instruments and execute over $300B in global trading volume: fully automated, with no discretionary human intervention. This domain combines massive data scale with high noise, adversarial dynamics, and frequent regime shifts, making it both scientifically challenging and commercially impactful. For machine learning researchers, it serves as a rigorous proving ground where advances in **time-series modeling, large-scale optimization, representation learning, and foundation models** can translate directly into measurable real-world outcomes. This talk will provide a high-level overview of our research agenda, infrastructure, and key open challenges at the intersection of large-scale AI and quantitative finance."

### 关键提取
- **四大技术方向**（演讲明确提及）：
  1. **Time-series modeling**（时间序列建模）
  2. **Large-scale optimization**（大规模优化）
  3. **Representation learning**（表示学习）
  4. **Foundation models**（基础模型）
- **无幻灯片/录像/全文公开**（截至检索时间）
- 同一演讲也在 **UMN CSE DSI** 举办（University of Minnesota），摘要略有不同：
  > "Every day, our systems generate **deep learning forecasts** for tens of thousands of instruments and trade over $300B globally - fully automated, no discretionary human input. This is a domain with massive data, high noise, and extreme regime shifts, making it both scientifically challenging and commercially rewarding. For ML researchers, it represents a proving ground..."

### 演讲者背景
- **Dr. Zhangyang "Atlas" Wang**
- XTX Markets **Research Director**，纽约 AI Lab 创始人兼负责人
- 原 UT Austin **Temple Foundation Endowed Associate Professor**（现休假）
- 学术荣誉：**IEEE AI's 10 to Watch**、**NSF CAREER Award**
- 专注：为金融时间序列与市场数据开发**大规模基础模型**

---

## 2. XTY Labs（XTX 的 ML 研究部门）

### 来源：FX Algonews + LiquidityFinder（2025 年报道）
- XTX 成立了 **XTY Labs**，一个专门的机器学习研究部门
- 由 Atlas Wang 领导
- 使命："rapidly turn the latest AI breakthroughs into tangible market advantages"
- 定位："the crucible of next-generation [ML solutions]"

### 基础设施数据（LiquidityFinder 报道，2025 年）
| 维度 | 数据 |
|------|------|
| 研究集群 | **100,000 cores** |
| GPU | **20,000 A100/V100 GPUs**（当时数据，后增至 25,000+） |
| 存储 | **390 PB** usable storage（后增至 650 PB） |
| RAM | **7.5 PB** |
| 日交易量 | **>$250B**（后增至 $300B+） |
| 员工 | 200+（伦敦、巴黎、纽约、孟买、埃里温、新加坡） |

### 来源：Lex Substack（2026 年）
> "XTX Markets has built one of the largest privately held GPU clusters in algorithmic trading, comprising over **25,000 GPUs—including 10,000 Nvidia A100s and 10,000 V100s**—paired with **650 petabytes** of high-speed storage."

---

## 3. XTX 招聘 JD 提取的技术栈

### 来源：Quant Blueprint（2026 年）
XTX 招聘 Quantitative Researcher 时要求的 ML 技能：

> "You need deep understanding of **supervised and unsupervised learning, deep learning, gradient boosting, Bayesian methods, and reinforcement learning**. XTX expects you to understand algorithms at a mathematical level — not just API calls."

> "XTX looks for candidates with deep machine learning expertise — not just familiarity with scikit-learn, but genuine understanding of modern ML methods including **deep learning, reinforcement learning, Bayesian methods, and causal inference**."

> "Publications in top ML venues (**NeurIPS, ICML, JMLR**) are a strong [plus]."

### 提取的算法/方法清单
| 类别 | 具体方法 |
|------|----------|
| **监督学习** | 深度学习（Deep Learning）、梯度提升（Gradient Boosting） |
| **无监督学习** | 表示学习（Representation Learning）、聚类 |
| **深度学习** | 神经网络（具体架构未公开，但"foundation model"暗示 Transformer 类） |
| **强化学习** | Reinforcement Learning（用于交易策略优化） |
| **贝叶斯方法** | Bayesian Methods（不确定性量化） |
| **因果推断** | Causal Inference（区分相关与因果） |
| **编程** | Python / C++ |

### 来源：XTX 官网 Careers 页
> "The quantitative research team is responsible for designing the **statistical models** that underpin our trading activities. Financial markets are data-rich, but the key lies in making sense of this data. We deploy **state-of-the-art machine learning techniques**, powered by extensive computational resources, to **sift through the noise and generate price forecasts** for a wide range of financial assets."

---

## 4. 行业背景：金融时间序列基础模型（Foundation Models）

### 来源：Jonathan Kinlay 博客（2026-02）
> "Time Series Foundation Models for Financial Markets: Kronos and the Rise of Pre-Trained Market Models"

- **Kronos**：一个金融时间序列基础模型，使用 **Transformer 架构**
- 将 K-line（OHLCV）作为 token 输入 Transformer
- 学习 P(K_{t+1:K} | K_{1:t})——给定历史 K 线，预测未来 K 线的概率分布
- 使用 **stacked self-attention layers**

### 来源：Nixtla（TimeGPT）
- **TimeGPT**：第一个公开的预训练时间序列基础模型
- 使用 **Transformer-based architecture**
- 输出：point forecasts + calibrated prediction intervals
- 支持 exogenous variables

### 来源：Medium（Jan Daniel Semrau）
- **TimeXer**：Transformer 用于时间序列预测，支持 exogenous variables
- 使用 **self-attention mechanism** 处理全局上下文
- 适用于金融市场的快速变化

---

## 5. 综合推断：XTX 可能使用的模型架构

**注意：以下为基于公开信息的合理推断，非官方确认。**

| 模型类型 | 证据 | 用途 |
|----------|------|------|
| **Transformer（Decoder-only）** | "Foundation models" + Atlas Wang 学术背景（视觉 Transformer） | 金融时间序列预测 |
| **LSTM/GRU** | 传统时间序列深度学习（XTX 招聘要求"deep learning"） | 短期价格预测 |
| **Gradient Boosting（XGBoost/LightGBM）** | 招聘 JD 明确提及"gradient boosting" | 特征重要性/非线性关系 |
| **Reinforcement Learning** | 招聘 JD 明确提及 | 交易策略优化（下单时机/大小） |
| **Bayesian Methods** | 招聘 JD 明确提及 | 不确定性量化（预测区间） |
| **Causal Inference** | 招聘 JD 明确提及 | 区分相关与因果，避免 spurious correlation |
| **Representation Learning** | 演讲摘要明确提及 | 从原始数据学习有意义的特征表示 |

---

## 6. 数据管道（推断）

### 输入数据
- **53,000+ 金融工具**的价格数据（OHLCV）
- **650 PB** 存储（历史数据 + 实时数据）
- 可能包括：订单簿数据、成交量、波动率、宏观指标、另类数据

### 处理流程（推断）
1. **数据收集**：从交易所/做市商/数据供应商获取原始数据
2. **特征工程**：从原始数据提取特征（收益率、波动率、动量、均值回归等）
3. **表示学习**：用深度学习从原始数据学习有意义的表示（而非手工特征）
4. **模型训练**：在 25,000+ GPU 上分布式训练
5. **推理**：实时生成价格预测
6. **执行**：全自动下单

### 输出
- **价格预测**（price forecasts）：对每个工具的未来价格方向/幅度
- **置信度**（uncertainty）：预测的不确定性（贝叶斯方法）
- **交易信号**：基于预测自动生成买卖信号

---

## 7. 主要来源

1. Stony Brook AI3 Seminar（Atlas Wang, 2026-03-09）: https://ai.stonybrook.edu/news/seminars/algorithmic-trading-large-scale-deep-learning
2. UMN CSE DSI Seminar（Atlas Wang）: https://cse.umn.edu/dsi/events/cse-dsi-machine-learning-seminar-atlas-wang-xtx-markets-ut-austin
3. FX Algonews（XTY Labs 成立）: https://fxalgonews.com/xtx-markets-aims-to-redefine-algo-trading-with-launch-of-xty-labs
4. LiquidityFinder（XTY Labs + 基础设施）: https://liquidityfinder.com/news/xtx-markets-launches-new-machine-learning-division-xty-labs-led-by-atlas-yang-c0033
5. Lex Substack（GPU 集群详情）: https://lex.substack.com/p/ai-the-secret-ai-supercomputers-powering
6. Quant Blueprint（XTX 招聘 JD）: https://www.quantblueprint.com/guides/how-to-get-a-job-at-xtx-markets
7. XTX Markets Careers: https://www.xtxmarkets.com/careers
8. Jonathan Kinlay（Kronos 基础模型）: https://jonathankinlay.com/2026/02/time-series-foundation-models-for-financial-markets-kronos-and-the-rise-of-pre-trained-market-models
9. Nixtla（TimeGPT）: https://www.nixtla.io/industries/financial-markets
10. Medium（TimeXer）: https://medium.com/@jsemrau/forecasting-real-time-market-data-with-transformer-fc8f96bd6b8e
