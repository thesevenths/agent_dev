# Step 2 — XTX Markets 机器学习预测方法（公开技术细节检索）

AS_OF: 2026-10-08 23:07（检索时间；以下为公开报道/学术/官网信息，非实时行情）

## 1. 公司级官方口径（xtxmarkets.com 官网）
> "We are a leading algorithmic trading firm that uses state-of-the-art machine learning technology to produce **price forecasts for over 53,000 financial instruments** across equities, fixed income, currencies, commodities and crypto. We use those forecasts to trade on exchanges and alternative trading venues, and to offer differentiated liquidity directly to clients worldwide."

- 技术页表述：拥有"trading industry 中无可匹敌的算力"，研究集群持续扩张；团队背景覆盖纯数学、编程、物理、计算机科学、机器学习。
- 正在芬兰建设大型数据中心以"future-proof"其算力。

## 2. 关键人物：AI Lab 负责人（重要新发现）
- **Dr. Zhangyang "Atlas" Wang** — XTX Markets **Research Director**，纽约 AI Lab 创始人兼负责人。
  - 专注：**为金融时间序列与市场数据开发大规模基础模型（large-scale foundation models）**，依托 XTX 自研 AI 基础设施。
  - 学术背景：原德州大学奥斯汀分校 Temple Foundation Endowed 副教授（现休假）。
  - 来源：Stony Brook AI Innovation Institute 研讨会公告（2026-03-09 演讲 "Algorithmic Trading with Large-Scale Deep Learning"）。
- 演讲摘要（关键）：
  > "At XTX Markets, we view algorithmic trading as one of the most compelling real-world frontiers for **deep learning and foundation models**. Every day, our systems generate forecasts for **tens of thousands of financial instruments** and execute over **$300B in global trading volume**: fully automated, with no discretionary human intervention. This domain combines massive data scale with high noise, adversarial dynamics, and frequent regime shifts..."
  - 要点：XTX 把算法交易定位为"深度学习与基础模型"的前沿应用场景；强调**大规模数据 + 高噪声 + 对抗性 + 频繁 regime shift** 四大挑战。

## 3. Gerko 本人关于 ML 方法的直接引语（Bloomberg 2025-01-22 访谈）
> "By building things ourselves, we can build ahead of our needs. **Because of the way we use machine learning to build our trading strategies, we have been confident that we can apply more compute power to ultimately generate better returns.**"

- 核心逻辑：**算力 → 更好的 ML 模型 → 更好的预测 → 更好的回报**（scale 假设成立）。
- 配套动作：自建 5 座芬兰数据中心（>€1B）、25,000+ GPU、650 PB 存储、新设 ML 研究员专职部门。

## 4. WSJ 2026-04-23 长文（付费墙，仅摘要可见）
- 标题："The Billionaire Math Geek Who Turned AI Into a Money-Printing Machine"
- 副标题："Alex Gerko's XTX Markets uses **Nvidia chips and 'deep learning'** to forecast price moves"
- 可见片段："Alex Gerko's edge as a trader comes from a **supercomputer powered by geothermal energy in Iceland**. Years before ChatGPT became a household name, his trading firm, XTX Markets, built an artificial-intelligence system aimed squarely at one goal: making money."
- 注意：WSJ 提到**冰岛地热超算**（与芬兰数据中心为不同设施/时期），说明 XTX 算力布局跨多地。
- 全文付费，技术细节（模型架构、特征、训练流程）未公开。

## 5. 与传统高频交易（HFT）的区别（综合 Bloomberg/WSJ/官网）
| 维度 | 传统 HFT | XTX Markets（ML 路线） |
|------|----------|------------------------|
| 核心优势 | 速度（低延迟、colocation） | **预测精度**（用算力换 alpha） |
| 信号来源 | 简单统计/微观结构信号 | **深度学习/基础模型**对 5 万+ 工具做价格预测 |
| 算力用途 | 加速执行 | **训练与推理大规模 ML 模型** |
| 基础设施 | 交易所机房 colocation | 自建数据中心（芬兰 5 座 + 冰岛超算） |
| 团队 | 工程师为主 | 纯数学/物理/CS/ML 研究员 |
| 华尔街质疑 | — | 数据有限且噪声大，深度学习能否真正预测市场存疑；"速度 vs 复杂度"权衡 |

## 6. 公开技术论文/学术产出
- **XTX 未公开具体模型架构论文**（截至检索时间）。
- Gerko 本人有数学博士论文及"市场择时"研究论文（Wikipedia 提及），但未公开 ML 交易模型细节。
- Atlas Wang 的学术背景（UT Austin）涉及深度学习/视觉，但其 XTX 内部工作未公开论文。
- Stony Brook 研讨会（2026-03-09）是**最接近公开技术细节**的渠道，但仅有摘要，无幻灯片/录像公开。

## 7. 可推断的技术栈（基于公开信息，非官方确认）
- **模型类型**：深度学习 + 基础模型（foundation models）用于金融时间序列（Atlas Wang 演讲明确）。
- **硬件**：Nvidia GPU（25,000+ 张）、自建数据中心（芬兰 Kajaani ×5 + 冰岛地热超算）。
- **数据规模**：53,000+ 金融工具、日交易量 >$300B、650 PB 存储。
- **训练范式**：大规模分布式训练（"apply more compute power to generate better returns" 暗示 scale law 思路）。
- **执行**：全自动、无人为干预（"fully automated, with no discretionary human intervention"）。
- **挑战**：高噪声、对抗性、频繁 regime shift（Atlas Wang 摘要）。

## 8. 主要来源
1. XTX Markets 官网: https://www.xtxmarkets.com
2. Stony Brook AI3 Seminar（Atlas Wang, 2026-03-09）: https://ai.stonybrook.edu/news/seminars/algorithmic-trading-large-scale-deep-learning
3. Bloomberg 2025-01-22（Gerko 访谈）: https://www.bloomberg.com/news/articles/2025-01-22/gerko-s-xtx-to-build-1-billion-data-hub-in-machine-learning-bet
4. Yahoo Finance / Bloomberg 转载: https://finance.yahoo.com/news/billionaire-alex-gerko-xtx-build-050000389.html
5. WSJ 2026-04-23（付费墙）: https://www.wsj.com/finance/alex-gerko-xtx-markets-ai-d155626a
6. Wikipedia — Alex Gerko: https://en.wikipedia.org/wiki/Alex_Gerko
7. paperswithbacktest.com: https://paperswithbacktest.com/course/alex-gerko
8. Bloomberg Business Facebook（2026-04-17，$3B 分红）: https://www.facebook.com/bloombergbusiness/posts/...

## 9. 结论（供 Step 3 综合）
- XTX 的 ML 方法**核心是"用大规模算力训练深度学习/基础模型来预测 5 万+ 金融工具的价格"**，而非传统 HFT 拼速度。
- 具体模型架构、特征工程、训练流程**未公开**（无论文、无技术博客、WSJ 付费墙）。
- 最接近公开技术细节的渠道：Atlas Wang 在 Stony Brook 的演讲（2026-03-09）+ Bloomberg 访谈 + WSJ 长文（付费）。
- 可确认的技术要素：深度学习 + 基础模型、Nvidia GPU 集群、自建数据中心、全自动执行、5 万+ 工具价格预测。
