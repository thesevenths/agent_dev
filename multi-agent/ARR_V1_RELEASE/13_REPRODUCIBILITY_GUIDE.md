# ARR v1.0 — 13 Reproducibility Guide

## 1. 导出与校验

```
py -m research_runtime.reproducibility --program-dir ARR_V1_EVIDENCE/program_e2e --out-dir <bundle>
```
输出 `REPRODUCIBILITY_MANIFEST.json`：每个文件 SHA-256、program/report manifest SHA、
verification（program_integrity / n_rounds / all_files_have_sha）。

正式包：`ARR_V1_EVIDENCE/reproducibility_package`（22 文件，manifest sha 6624f8ce...，integrity true）。

## 2. 复现步骤

1. checkout 本分支（见 ARR_V1_VERDICT.json 的 final_commit）。
2. 确认 `config.py`/`llm.py` 的用户改动（如需要）从备份 patch 恢复。
3. `py -m pytest -q research_runtime/tests` → 869 passed / 12 skipped。
4. 真实生产 E2E 需：vLLM 凭证 + OpenAlex 可达 + 真人带外审批（见 M7_6_REAL_E2E_REPORT.md）。
5. 多轮 program：`py -m research_runtime.run_arr_v1_program_e2e --rounds 3`（真实 LLM）。

## 3. 不伪造原则
所有 SHA 从磁盘重算；无模型生成的 SHA 被信任；无外部依赖时不产出结果，标 BLOCKED。
