# 步骤1：XTX Markets 及 Alex Gerko 机器学习量化策略公开信息检索

AS_OF: 2026-10-09 09:09（检索时间；以下为公开报道/官网/招聘/学术资料，非实时行情数据）

> 说明：本步骤仅做公开信息检索与整理，供下游步骤（盈利逻辑分析、个人模仿可行性评估）使用。
> 所有信息均来自公开来源，已标注出处。XTX 的核心模型细节属商业机密，公开渠道只能看到"技术路线 + 基础设施 + 招聘描述"层面的信息。

---

## 一、公司与创始人背景（事实基础）

- **XTX Markets**：英国伦敦算法交易公司，2015年1月由 Alexander (Alex) Gerko 创立，是 GSA Capital 的衍生（spin-off）。Gerko 持股约 75%，现任 co-CEO（与 Hans Buehler 共同）。
- **规模**：约 300 名员工（伦敦、新加坡、纽约、巴黎、布里斯托尔、孟买、埃里温、芬兰 Kajaani）；日交易量约 $250bn–$300bn，覆盖 35 个国家。
- **盈利记录**：2022 年利润 £1.1bn（同比 +64%）；2023 年营收约 £2.0bn、净利约 £835m；2024 年 Gerko 从 XTX 获得 £683m（公司预留约 £1.28bn 分给创始人及 30 名量化交易员）；2025 年报道利润约 £895m。
- **Gerko 背景**：俄罗斯出生、英国籍，数学博士（莫斯科国立大学），曾在 Deutsche Bank 做 FX 交易、GSA Capital 任 FX 交易主管。写过市场择时（market timing）研究论文。2026 年身家约 $17.1bn（Bloomberg/Forbes）。
- 来源：Wikipedia (XTX Markets / Alex Gerko)、Forbes、Bloomberg、Jawlah。

---

## 二、问题1：用了哪些机器学习算法？预测了哪些价格？

### 2.1 预测目标（预测什么）
- **核心是"价格预测"（price forecasts）**：官网明确表述——"用 state-of-the-art machine learning 技术，为 **53,000+ 金融工具** 生成价格预测，覆盖 **股票、固定收益、外汇、大宗商品、加密货币**"。
- 这些预测被用于：(a) 在交易所/替代交易场所直接交易；(b) 向客户提供流动性（做市/流动性提供）。
- 学术演讲（Atlas Wang, Research Director, XTX AI Lab, Stony Brook 2026-03-09）进一步说明：系统每天为"数万金融工具"生成预测，执行 $300bn+ 全球交易量，**全自动、无人为裁量**；领域特点是"海量数据 + 高噪声 + 对抗性动态 + 频繁 regime shift（市场状态切换）"。
- 输入数据（据 Lex Substack 报道）：模型每天处理 **超过 1 万亿个数据点**，包括 **订单簿（order book）、tick 数据、情绪（sentiment）信号** 等。

### 2.2 算法演进路线（关键：从简单到深度）
XTX 官方招聘描述（Quantitative Researcher - Deep Learning, Built In）给出了最权威的技术演进：
> "过去十年，我们的模型从赋予公司名字（XTX = eXtreme Trading eXchange / econometric）的**计量经济学方法（econometric methods）**，演进到**树模型（trees，如 XGBoost/LightGBM 类）**和**神经网络（neural networks）**，再到**现代深度学习（modern deep learning）**。我们期望这一演进继续。"

- 即技术栈演进：**计量经济学 → 树模型（GBDT 类）→ 神经网络 → 现代深度学习 / 大规模 foundation models（基础模型）**。
- 最新方向（Atlas Wang 演讲标题）："Algorithmic Trading with **Large-Scale Deep Learning**"，AI Lab 专注开发**面向金融时间序列和市场数据的"大规模基础模型（foundation models）"**。
- 注意：XTX 并未公开点名具体网络结构（如是否用 Transformer/LSTM），公开层面只到"deep learning / foundation models"这一层级。

### 2.3 基础设施（这是"押注 ML"的物质基础）
- **25,000+ GPU** 研究集群（含约 10,000 张 Nvidia A100 + 10,000 张 V100）。
- **1 Exabyte+ 可用存储**（另有报道提到 650PB 高速存储）。
- 在**芬兰 Kajaani 投资 €1bn 建 5 个数据中心**（利用当地凉爽气候 + 地热/低成本电力，Kajaani 已有欧洲 LUMI 超算）。
- 早期在**冰岛**用**地热能源**驱动的超算（WSJ 报道）。
- 设立 **XTX Labs / AI Lab**（纽约）+ **AI Residency Program**（AI 研究员月薪 $40k–$50k），专门做"金融 × 机器学习"研究。
- 来源：XTX 官网、Bloomberg、WSJ、Built In 招聘、Stony Brook 学术演讲、Lex Substack。

---

## 三、问题2：ML 算法都公开，谁都能用，为什么 XTX 能赚钱？

公开信息能支撑的"护城河"（非 XTX 独家声明，而是从报道/招聘/基础设施推断）：

1. **算力与数据规模（最核心）**：25,000+ GPU、1EB 存储、€1bn 数据中心。Gerko 原话（Bloomberg 访谈）："**By building things ourselves, we can build ahead of our needs... we have been confident that we can apply more compute power to ultimately generate better returns.**"（自建基础设施，用更多算力换更高回报。）——算法公开，但"用得起、用得上"这种算力的人极少。
2. **数据优势**：每天处理 1 万亿+ 数据点（订单簿、tick、情绪信号）。原始高频/订单流数据的采集、清洗、存储本身就是巨大壁垒。
3. **执行与成本工程**：proprietary 技术优化交易执行、最小化滑点和交易成本（paperswithbacktest 描述）。ML 预测的边际优势很小，能否落地为利润取决于执行速度和成本。
4. **研究人才密度**：300 人中聚集纯数学、物理、CS、ML 背景的研究者；专门 AI Lab + 高薪酬 AI Residency。
5. **规模化的"小价差"套利**：靠 AI 检测市场中的微小价差，通过**数百万笔自动化交易**累积（Jawlah 描述）。单笔利润极小，靠规模和速度取胜。
6. **流动性提供/做市角色**：不仅是自营，还向机构客户提供流动性，形成收入多元化。
7. **持续演进的研究文化**：从计量→树→NN→deep learning→foundation models 的持续迭代，"good ideas should not be bottlenecked by data, infrastructure or compute"。

> 关键结论（供下游步骤展开）：**XTX 的盈利不来自"某个独家算法"，而来自"算力 + 数据 + 执行 + 人才 + 工程"的系统性规模优势。** 算法本身公开，但把公开算法在 53,000 个工具上、用 25,000 GPU、以极低延迟和成本落地并持续迭代，是个人几乎无法复制的。

---

## 四、来源清单（URL）

1. https://en.wikipedia.org/wiki/XTX_Markets
2. https://en.wikipedia.org/wiki/Alex_Gerko
3. https://www.xtxmarkets.com （官网：53,000+ 工具、25,000+ GPU、1EB 存储、$250bn 日交易量）
4. https://builtin.com/job/quantitative-researcher-deep-learning/9716220 （技术演进：econometric→trees→NN→deep learning）
5. https://finance.yahoo.com/news/billionaire-alex-gerko-xtx-build-050000389.html （Bloomberg：€1bn 芬兰数据中心、25,000 GPU、Gerko 访谈原话）
6. https://www.wsj.com/finance/alex-gerko-xtx-markets-ai-d155626a （WSJ：Nvidia 芯片 + deep learning 预测价格、冰岛地热超算）
7. https://ai.stonybrook.edu/news/seminars/algorithmic-trading-large-scale-deep-learning （Atlas Wang 演讲：foundation models、$300bn 全自动交易）
8. https://lex.substack.com/p/ai-the-secret-ai-supercomputers-powering （25,000 GPU 构成、1 万亿数据点/日、订单簿/tick/情绪信号）
9. https://www.forbes.com/profile/alexander-gerko （身家、背景、分红）
10. https://jawlah.co/en/63361 （2025 利润 £895m、微小价差 + 数百万笔自动交易）
11. https://paperswithbacktest.com/course/alex-gerko （数据驱动策略、执行优化、最小化滑点）
12. https://community.portfolio123.com/t/xtx-markets-unveils-machine-learning-lab-blending-finance-with-technology/66200 （XTX Labs / AI Residency Program）

---

## 五、信息缺口（供下游步骤注意）

- XTX **未公开**具体网络架构（是否 Transformer/LSTM/特定 foundation model 结构）、具体特征工程、回测细节、夏普/容量数据——这些是商业机密。
- "预测精度"的具体数字（如方向预测准确率、IC 值）公开渠道不可得。
- 下游步骤做"个人模仿可行性"评估时，应基于"公开算法 + 公开基础设施差距"来论证，而非虚构 XTX 内部细节。
