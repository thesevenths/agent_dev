[Most recent upstream result] [Most recent upstream result] [Most recent upstream result] Step 2 完成。已检索金融 ML 基础模型行业实践。

**AS_OF: 2026-10-08 23:16**（检索时间；以下为公开研究/论文/行业报告，非实时行情）

**核心发现：**

1. **通用 TSFM 在金融数据上表现极差**
   - **TimesFM**（Google, 500M 参数）：金融零样本预测 **R² = -2.80%**（负值=比均值预测还差）
   - **Chronos**（Amazon, 710M 参数）：金融零样本预测 **R² = -1.37%**（同样无效）
   - **原因**：金融数据具有**高噪声、对抗性、频繁 regime shift**，与零售/制造/医疗等场景差异巨大
   - **含义**：XTX 的"foundation model"必须是**金融专用**，而非直接套用通用 TSFM

2. **金融专用 TSFM（Domain-Specific）**
   - **Kronos**（AAAI 2026）：Transformer 架构，输入 K-line（OHLCV），输出未来 K 线概率分布
   - **FinCast**：Transformer-based，支持多变量（宏观指标）
   - **关键结论**：2024-2025 年 TSFM 生态分化为两条路线——通用型（Google/Amazon/Salesforce）vs. 领域专用型（Kronos/FinCast），后者在金融领域表现优于前者

3. **LLM + Transformer 混合架构（学术前沿）**
   - **LLM-Transformer 混合**：LLM 提取新闻情绪 + Transformer 建模价格动态
   - **LLM 生成 Formulaic Alpha**：LLM 自动生成数学表达式（alpha 因子），自动化、可扩展
   - **LLM-Augmented Linear Transformer-CNN**：LLM 作为"专业金融分析师"执行技术分析
   - **对冲基金视角综述**（arXiv:2605.05211）：LLM 在量化金融中的应用包括情绪提取、财报分析、价格序列 tokenization、多智能体交易系统

4. **XTX 方法的行业定位**
   | 维度 | 通用 TSFM | 金融专用 TSFM | XTX |
   |------|-----------|---------------|-----|
   | **设计目标** | 跨领域泛化 | 金融专用 | **金融专用 + 实盘验证** |
   | **金融表现** | R² = -2.80%（无效） | 优于通用 | **未公开，但日交易量 >$300B** |
   | **算力** | 单 GPU | 学术级 | **25,000+ GPU** |
   | **执行** | 离线预测 | 研究/演示 | **全自动实时交易** |
   | **验证** | 回测 | 回测 | **实盘 P&L** |

5. **XTX 的独特性**
   - **规模**：25,000+ GPU、650 PB 存储——远超学术/创业公司
   - **实盘验证**：日交易量 >$300B，非回测
   - **全自动**：无人工干预，纯模型驱动
   - **保密**：无公开论文/代码，技术细节完全保密
   - **金融专用**：非通用 TSFM，而是为金融数据专门设计

**完整报告已保存至：** `F:\agent\multi-agent\tmp\step2_financial_ml_foundation_models.md`