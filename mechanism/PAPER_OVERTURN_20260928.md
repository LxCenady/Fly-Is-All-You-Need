# 论文稿 `E:\flybrain_e5a_v3_20260920\main.tex` 推翻清单（2026-09-28）

原稿 91,532 B / `8432B67D…`，**未改动**（它是被引对象）。修订稿另存。
证据：`D:\苍蝇。\审核_20260928\`（CSR bug）、`D:\苍蝇。\mechanism\RESULTS_20260928.md`（重跑、M1、M2、KC 密度、记忆曲线）以及下文列出的代码行。

判定分四类：
- **作废**：数据来自修复前的模拟器，数字本身不成立。
- **恒真**：由代码结构保证成立，不是测量结果。
- **误读**：数字还在，但解释被新机制结果推翻。
- **保留**：仍然成立，需按新口径限定。

## A. 作废（修复前数据；资产 A 早于 09-17 22:56 的 CSR 修复）

| 位置 | 原主张 | 新事实 |
|---|---|---|
| 标题 | "…Decodable in the KC→MBON Weight State but Buys No Measurable Readout Advantage" | "没有增益"建立在错误写入上。修复后，可塑性让 activity 容量**下降**约 0.04（3/3 seed），突触状态则带有输入窗口之外的历史 |
| 摘要第 2–3 句；§1.2；§Results "Capacity dissociation"；Table `tab:capacity` M2 / M3a / M3b / M4 | frozen 1.17 vs plastic 1.16，"indistinguishable"；weight 0.23、edge 0.36，均为平坦剖面 | 修复后：frozen 1.162–1.186（复现），plastic **1.126–1.131**；edge sketch 0.43–0.46；weight 0.16–0.26。旧 M2–M4 行全部作废 |
| 图 1 标题 | "decodable memory is a property of neural activity, not of the plastic weight state" | 方向相反：lag 4–12（输入窗口之外）上，edge sketch 为 0.11–0.16，neural 只有 0.004–0.028（噪声偏置量级） |
| §Discussion 第三条特征；预测 P3 | "plastic 1.16 vs frozen 1.17"；"plasticity's contribution to readout capacity is near zero" | 作废，见上 |
| §Limitations "Measurement assets"；§What This Paper Does Not Claim 中引用 0.36 / 0.23 的各条；§Data 中资产 A 为容量来源 | — | 资产 A 的可塑组数字全部作废；frozen 组数字保留 |

## B. 恒真（由代码结构保证，不是发现）

| 位置 | 原主张 | 为什么恒真 |
|---|---|---|
| 摘要"no-write readout geometry is invariant to plasticity dose and delay"；§Results "item geometry is fixed and dose-invariant"；图 2；Discussion 中 P1 "confirmed" | f1(R) 在 A0 / A2 / A4 × 三种延迟上一致到 2.6e-9 | R 是 **no-write** 分支的响应：`_episode(..., allow_plastic=False)` 下不发生任何写入（`mb_persistent_memory_probe.py:630`），η 与 τ 对它不起作用，所以"对剂量不变"是恒真的。延迟 32 / 72 / 152 token 的间隔期网络完全静默；M1 实测快状态差在 32 token 时已降到 float32 分辨率，所以"对延迟不变"也是恒真的。论文自己承认 no-write 数组跨臂差 ≤ 2.4e-7。**P1 不是被确认，而是没有被检验** |
| 图 5；§Results "delay change … cosine 1.000000000000, norm ratio 2.4598" | 间隔只缩放写入状态、不旋转它，"which is what a pure scalar decay would do" | 间隔期可塑性关闭，调用的是 `decay_only`：`modulation *= _weight_decay`（`mb_plasticity.py:187`），这本身就是纯标量乘法。旋转在代码上不可能发生 |
| 摘要、§1.4、§Results "content reaches the readout"：1.711e6 倍 | 内容效应 0.0753 是地板 4.4e-8 的 1.7e6 倍 | 地板是 GPU 归约顺序噪声。任何非零写入都会得到 10⁵–10⁷ 倍的比值。这个数只能排除"严格等于零"，没有信息量 |
| 旧"持久记忆线"：15.36 s 后保留 38.3%（已不在本稿，但 INDEX 与旧文档有引用） | — | 38.3% = exp(−15.36/16)，就是手设常数。修复后曲线（§6）：关闭衰减时读出 123 s 内不变，遗忘 100% 来自手设衰减 |

## C. 误读（数字在，解释被新机制结果推翻）

| 位置 | 原解释 | 新机制事实 |
|---|---|---|
| 摘要最后一句；§Results "Factorizability"；Discussion 中"connectome supplies the readout geometry" | 写入效应方向由探针决定、与内容无关；内容只贡献一个小标量；几何由连接组提供 | 标准工作点的输入码是 **512 / 675 个 PN（76%）**。探针时 KC 激活约 87%，不同字符的 KC 集合 Jaccard 为 **0.93**。每个探针读到的几乎是同一批 KC，所以方向只能随探针的 MBON 响应模式变化。不同上下文的写入状态本身就近乎相同（论文自报 cos 0.976）。M2：内容由"每个 MBON 收到多少写入"承载，而非"写在哪些 KC"。**这是输入编码与工作点的性质，不是连接组的性质** |
| §Results "item geometry"：六个探针响应近共线（cos 0.9745–0.9932） | "measurement fact about this probe set" | 同一机制：稠密 KC 码使所有探针激活几乎同一批 KC |
| 图 5："direction of the write state is fixed by the first written letter" | 方向由第一个写入字母决定 | 不同字母写入的 KC 集合写入期 Jaccard 为 0.55–0.67，所以第一个字母写下的方向与后续字母几乎重合 |
| §Methods "Delayed-recall task"；图 1；Table `tab:capacity` M1 / B1 | neural 通道的"记忆"在两个字符内降到 chance | 编码器 `window=4, position_codes=True` 在每个 token 同时注入 lag 0–3，所以 lag 1–3 是输入回声，不是网络记忆。frozen neural 在 lag 4–12 上与噪声偏置同量级：**活动态没有超出输入窗口的记忆**。资产 B 的 lag 1–3（0.98 / 0.72 / 0.20）同理 |
| §Methods 与 §Results：neural 读出 "at 4,096 units" 被当作"读出" | — | 4096 维 = 全部 4,064 个 KC + 97 个 MBON 中的前 32 个。KC 在可塑突触的**上游**。"没有读出增益"主要测的是 KC 层 |
| §Discussion 理论框架（Dong 2020 等）："frozen substrate supplies geometry, write is a slow variable the geometry does not convert into item-specific readout"；§1.3 "an edge-specific write is not readable as item identity at the population level in this model" | — | **被推翻**：在 PN 码 160–224 的稀疏区，读出变成线索特异的（本项探针是其他探针的 1.5–29 倍）；打乱 KC 身份会让本项读出只剩 13–30%（`m2_current.json`）。"不能按项目读出"是稠密输入工作点的性质，不是连接组的性质 |

## D. 保留（修复后数据，按新口径限定）

| 位置 | 状态 |
|---|---|
| §Results "gain mechanism is rejected"（E5A v2，09-19，修复后） | 保留：指定的"探针自身响应缩放"模型被拒绝，6/6。需补一句：它在稠密 KC 区测得，稀疏区未测 |
| §Results attenuation 表（E5A v2 数据） | 数字保留，是描述性量；表标题写成 "Asset A" 是**标注错误**，数据来自 E5A v2 的 WEIGHT_STATE 报告 |
| 统计单位合同、四上下文 F1_UNFOLD 的描述值（E4B，09-18，修复后） | 数值保留；解释改为 C 类，形式判决仍不许可 |
| frozen 容量 1.1664 与资产 B 2.905 | 保留（不受 bug 影响），但必须写明 lag 0–3 由输入注入 |
| §"What This Paper Does Not Claim" 的大部分条目 | 保留；引用 0.36 / 0.23 的条目删除 |

## 修订后的主线（替代原标题的论证）
1. 在秒级尺度上，记忆内容完全由 KC→MBON 权重承载；活动态在约 0.1 s 膜时间常数内消失，没有回响（M1）。
2. 保持完全来自手设衰减；网络动力学对遗忘与巩固的贡献为零；读出把指数迹变成阈值化的台阶（记忆曲线）。
3. 在标准工作点，KC 码稠密，MB 退化为"每个 MBON 一个标量存储"；内容靠 MBON 路由承载，不靠 KC 身份（M2 + KC 密度）。
4. 稀疏 KC 区存在（PN 码 160–192，探针期 KC 激活 9–16%）。在那里出现两层寻址：MBON 路由决定输出方向（所有工作点都成立）；KC 身份决定哪个探针能读出（只在稀疏区成立）。限定：3 对写入、一个臂；稀疏区的读出停在 MBON 阈下电压，没有传到下游。
4b. （2026-09-28 补充，6 对写入）线索特异性中位数：稀疏区 3.6–4.7，稠密区 1.06。KC 身份寻址在 6/6 对上成立。随机 PN→KC 接线可复现稀疏化与寻址；真实接线的特殊性在随机输入码下无法检验。下游只经 MBON 放电翻转接收记忆（RESULTS §8）。
4c. （M3，RESULTS §9）项目输出的区分度来自**真实的 KC→MBON 路由**：项目间输出余弦真实为 0.76，路由置乱后为 0.96–0.98。这是第一个依赖真实连接结构的结果。PN→KC 接线的具体结构对稀疏化不必要。
4d. （RESULTS §10）下游只有在 MBON 放电栅格改变时才能看到记忆（35 / 35 对比 0 / 93）；路由效应主要来自比注释 KC 亚型更细的个体接线；PN→KC 接线可能带来的线索特异性增益，按事先规则不成立。
4e. （RESULTS §11）输出的项目身份需要 PN→KC 与 KC→MBON 两层真实接线的配对：随机化任意一层都会让项目输出几乎一样（0.97–0.99）。整套交换 KC 输出档案也不能恢复。新论文稿：`D:\苍蝇。\paper_v2\main.tex`。
4f. **（RESULTS §12，修正 4c–4e）** 输出身份只需要两层接线的**粗粒度**结构，即半脑与 KC 亚型：同侧同亚型内任意交换 KC 输出档案，区分度完全保留。随机码下，区分度主要来自阈值放大造成的偏侧存储；双侧对称码下，来自各项目招募的 KC 亚型组成不同。多巴胺门控不携带项目身份。
5. 原稿之外、INDEX 中的 S1–S3 / R2 / S5 方向因子分解那条线，也全部跑在 512 PN 的稠密工作点上。它们的形式判决本来就不许可；新结果说明，即便许可，它们描述的也是输入编码造成的退化，而不是 MB 连接结构的作用。**不建议再投入。**
