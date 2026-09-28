# 可疑点审核（2026-09-28）

只读审核，未改动任何冻结件或产物。复现：`D:\flybrain_lm_cuda` 下以 `PYTHONPATH=D:\flybrain_lm_cuda` 运行两个脚本（GPU 脚本需 sitecustomize）。

## F1（严重）资产 A 与整条持久记忆线早于 CSR 规范化修复
- `flybrain/brain.py:110-117` 的 `sum_duplicates()+sort_indices()` 修复写于 2026-09-17 22:56；资产 A（`mb_plasticity_vocab65_biological_formal.json`）写于 09-16 21:23；`mb_persistent_memory_probe_*`、`mb_capacity_*`、`mb_long_gap_curve_tau16` 等均在 09-17 22:56 之前。E4B / E5A / dose4 均在修复之后。
- `csr_check.json`：`sensory_input=False` 行掩码后 148,687 行实际未排序；按修复前路径缓存的 61,210 个 KC→MBON 槽位，规范排序后 **61,164 个**指向不同的突触前细胞，其中 **11,130 个**不是 KC，**3,725 个**符号与缓存 w0 相反；仅把 w0 写回（modulation=0）就使 97 个 MBON 的输入行相对 L2 改变 **63.4%**。
- `gpu_sort_check.json`：CuPy 14.2.0 上第一次 SpMV **原地**把 CSR 规范化（indices 哈希变为与 CPU 规范序一致）。
- 09-17 的 `deepseek_sp2_smoke_canonical_review.md` §10/§12 已把"历史产物是否受影响"列为【未知】且未重跑；此后文档与 `main.tex` 仍把资产 A 的 1.1581 / 0.228 / 0.356 当作"已确认"的主张。
- 未能按字节验证 09-16 版 `mb_plasticity.py`（后来改过）；结论建立在"缓存位置逻辑未变"这一前提上（修复说明只提到新增身份校验）。

## F2（设计）headline 的 `neural@4096` 基本不含可塑性的下游
- 资产 A：KC 4,064 + MBON 97 = 4,161 维；最大 size 为 4096 ⇒ 读出 = 全部 KC + 97 个 MBON 中的前 32 个。KC 是可塑突触的**上游**。

## F3（解释）lag 1–3 是输入编码直接注入的
- `window=4, gamma=0.5, position_codes=True`：每个 token 同时注入 lag 0..3 的独立位置码。frozen neural@4096 的 1.1664 中，lag 0 约占 0.959；lag 4–12 的 accuracy 在 chance 附近（测试集 n=897，SE≈0.004），其 `max(excess,0)` 贡献约 0.019，已大于 frozen 与 plastic 之差 0.0083。

## F4（次要）
- `INDEX.md` 第 131 行打印的复审件 sha256 只有 61 字符（实测 `63467e19…c5d84ddb1b152d3f7ec991fd`）。
- `INDEX.md` §5.4 仍写 runner = 165,393 B / `AA39439F…`（"实际执行版本"）；该路径现为 241,221 B / `6ADD2C40…`。
- A0 的 Δr 在 seed 之间的相对散布为 1.9 / 2.6（数值尘埃），P2 / P3 中涉及 A0 的方向比较，比的是浮点噪声。

## 已抽查且吻合
资产 A/B JSON、main.tex、口径真源、R2 json、S3 npz（22 B）、并集结果、复审件：字节数与哈希均与 INDEX 一致。
