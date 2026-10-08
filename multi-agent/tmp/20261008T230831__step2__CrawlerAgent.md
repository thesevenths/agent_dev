[Most recent upstream result] Step 2 完成。已检索 XTX Markets 机器学习预测方法的公开技术细节。

**AS_OF: 2026-10-08 23:07**（检索时间；以下为公开报道/学术/官网信息，非实时行情）

**核心发现：**

1. **公司官方口径（xtxmarkets.com）**
   - 用"state-of-the-art ML"对 **53,000+ 金融工具**做**价格预测（price forecasts）**，再据此在交易所/替代交易场所交易
   - 团队背景：纯数学、编程、物理、CS、ML

2. **关键人物：AI Lab 负责人（新发现）**
   - **Dr. Zhangyang "Atlas" Wang** — XTX Research Director，纽约 AI Lab 负责人
   - 专注：**为金融时间序列开发大规模基础模型（foundation models）**
   - 原 UT Austin 副教授，2026-03-09 在 Stony Brook 做演讲 "Algorithmic Trading with Large-Scale Deep Learning"
   - 演讲摘要要点：XTX 把算法交易定位为"深度学习与基础模型"前沿；强调**大规模数据 + 高噪声 + 对抗性 + 频繁 regime shift** 四大挑战；日交易量 >$300B，**全自动、无人为干预**

3. **Gerko 直接引语（Bloomberg 2025-01-22）**
   > "Because of the way we use machine learning to build our trading strategies, we have been confident that **we can apply more compute power to ultimately generate better returns**."
   - 核心逻辑：**算力 → 更好的 ML 模型 → 更好的预测 → 更好的回报**（scale 假设）

4. **WSJ 2026-04-23 长文（付费墙，仅摘要）**
   - 副标题："uses **Nvidia chips and 'deep learning'** to forecast price moves"
   - 提到**冰岛地热超算**（与芬兰数据中心为不同设施）
   - 全文付费，技术细节未公开

5. **与传统 HFT 的区别**
   | 维度 | 传统 HFT | XTX（ML 路线） |
   |------|----------|----------------|
   | 核心优势 | 速度 | **预测精度** |
   | 信号 | 简单统计/微观结构 | **深度学习/基础模型** |
   | 算力用途 | 加速执行 | **训练/推理大规模 ML 模型** |
   | 基础设施 | colocation | 自建数据中心（芬兰×5 + 冰岛超算） |

6. **具体模型架构/特征工程/训练流程：未公开**
   - XTX 无公开技术论文、无技术博客
   - 最接近公开技术细节的渠道：Atlas Wang Stony Brook 演讲（仅摘要）+ Bloomberg 访谈 + WSJ 付费长文
   - 可确认技术要素：深度学习 + 基础模型、25,000+ Nvidia GPU、自建数据中心、全自动执行、5 万+ 工具价格预测

**文件路径：** `F:\agent\multi-agent\tmp\step2_xtx_ml_method.md`

**留给 Step 3：** 综合 Step 1（背景）+ Step 2（ML 方法）回答用户问题；"苏联数学世家"说法未获英文公开资料佐证，需标注为"未证实"。