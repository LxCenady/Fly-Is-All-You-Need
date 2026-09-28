# Literature index — Fly-Is-All-You-Need

Updated 2026-09-28. DOIs of A2, D1, E1, F1 and G1 were checked against Crossref
on 2026-09-28 (title, venue, volume/pages, authors); all matched. D1's abstract
confirms MBON-α3 and 948 presynaptic KCs. The dataset used by the model is
MaleCNS v1.0 (FlyEM; synapse confidence ≥ 0.5), as downloaded by
`flybrain/build.py`, so G1 is its primary citation.

## A. Whole-brain / connectome-constrained computation
- **A1** Shiu PK, Sterne GR, Spiller N, et al. (2024). A *Drosophila* computational brain model reveals sensorimotor processing. *Nature* 634, 210–219. doi:10.1038/s41586-024-07763-9. — Executable whole-brain LIF connectome; closest methodological precedent. We study plasticity, storage, retrieval, routing and downstream transmission rather than frozen sensorimotor processing.
- **A2** Li Q, Ping W, Zhang K, Wang C. (2026). Connectome-constrained modeling identifies neurons and synapses that sustain spontaneous activity in *Drosophila*. bioRxiv, posted 2026-08-25. doi:10.64898/2026.08.21.745055. — FlyWire model fitted to calcium data, in-silico perturbation; target is spontaneous activity, not memory.

## B. Mushroom-body connectome / anatomy
- **B1** Takemura S, Aso Y, Hige T, et al. (2017). A connectome of a learning and memory center in the adult *Drosophila* brain. *eLife* 6:e26975. doi:10.7554/eLife.26975.
- **B2** Li F, Lindsey JW, Marin EC, et al. (2020). The connectome of the adult *Drosophila* mushroom body provides insights into function. *eLife* 9:e62576. doi:10.7554/eLife.62576. — Reference for claims about coarse topology vs individual synapse identity.

## C. Sparse KC coding / addressability
- **C1** Lin AC, Bygrave AM, de Calignon A, Lee T, Miesenböck G. (2014). Sparse, decorrelated odor coding in the mushroom body enhances learned odor discrimination. *Nat Neurosci* 17, 559–568. doi:10.1038/nn.3660. — Proposes sparse KC populations as precisely addressable locations. **Novelty boundary:** we do not claim that sparse coding enables memory specificity in general.

## D. KC→MBON memory module
- **D1** Hafez OA, Escribano B, Ziegler RL, Hirtz JJ, Niebur E, Pielage J. (2023). The cellular architecture of memory modules in *Drosophila* supports stochastic input integration. *eLife* 12:e77578. doi:10.7554/eLife.77578. — Biophysical MBON-α3 with 948 KCs; our MBONs are point neurons (limitation for the spike-gating result).

## E. DAN-gated plasticity / learning theory
- **E1** Bennett JEM, Philippides A, Nowotny T. (2021). Learning with reinforcement prediction errors in a model of the *Drosophila* mushroom body. *Nat Commun* 12:2569. doi:10.1038/s41467-021-22592-4. — DAN-gated KC→MBON plasticity is not new to this project.

## F. Memory dynamics / DAN / connectome-constrained model
- **F1** Huang C, Luo J, Woo SJ, et al. (2024). Dopamine-mediated interactions between short- and long-term memory dynamics. *Nature* 634, 1141–1149. doi:10.1038/s41586-024-07819-w. — Our retention is a prescribed decay; STM/LTM dynamics are absent (limitation).

## G. Male CNS connectome dataset
- **G1** Berg S, et al. (144 authors) (2026). Sexual dimorphism in the complete *Drosophila* male central nervous system connectome. *Cell* 189(18), 5504–5526.e15. doi:10.1016/j.cell.2026.08.015. — Primary dataset citation (replaces the former TODO in Methods).

## Novelty framing
Known: sparse KC coding and addressability (C1); KC→MBON architecture (B1, B2, D1); DAN-gated plasticity (E1 and experiments); executable connectomes (A1); connectome-constrained MB memory dynamics (F1); in-silico causal perturbation of whole-brain models (A1, A2).

Contribution of this project: in one executable whole-CNS MaleCNS model, a
systematic causal decomposition of a memory into storage substrate
(KC→MBON weights), retrieval address (KC identity, sparse regime only), output
routing (hemisphere and KC subtype), write location (DAN identity), lifetime
(prescribed decay), interference (KC overlap) and downstream readability (MBON
spike flips), with the conditions under which each operates.

Do not frame as: first *Drosophila* memory model; first connectome-constrained
fly model; discovery of sparse coding, of KC→MBON plasticity, or of dopamine
control of MB memory.
