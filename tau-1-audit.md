# Tau-1 Audit Report: Reasons for Suboptimal Model Performance

**Date:** November 29, 2025

## 1. Executive Summary
This report details initial findings regarding the suboptimal performance of SGLang-trained Qwen models in the Tau1 (Airline) environment. Key issues identified include a misalignment between the SFT training data and the environment's available tools, leading to reduced effective training data volume and potentially hindering robust tool-usage policy development. The inherent sparse reward structure of the benchmark also presents a challenge for RL fine-tuning.

## 2. Key Findings

### 2.1 Data Quality & Environment Alignment
*   **Tool Mismatch Resolution & Data Recovery:** Initially, the provided "Eigen AI" SFT dataset (derived from `tau_airline_mt_dialogs_*.jsonl`) frequently referenced a tool `get_flight_status` that was not defined or available in `tau-bench` (v0.1.0).
    *   **Action Taken:** The `get_flight_status` tool has been **implemented and integrated** into the `tau-bench` Airline environment.
    *   **Impact:** This resolved the tool mismatch. The entire SFT dataset of **287 conversations** (previously reduced to 64 due to strict filtering) is now fully aligned with the environment and available for training. This significantly enhances the potential for robust SFT grounding.
*   **Previous Limitation (Mitigated):** The issue of "Reduced Effective SFT Data Volume" has been fully mitigated by the tool implementation.

### 2.2 Reward Structure
*   **Sparse, Terminal Rewards:** The `tau-bench` environment provides a sparse reward signal (0.0 for all intermediate steps, and 1.0 for task success or 0.0 for failure only upon termination). This aligns with the initial observation of "sparse rewards" and makes reinforcement learning more challenging due to delayed feedback.

### 2.3 Tool-Usage Policy Development
*   **Potential Suboptimal Trajectory Format:** Observations from the raw SFT dataset show `[ASSISTANT]: <function>...</function>` immediately followed by `[ASSISTANT_THINKING]: ...`. This implies that the model might be generating the tool call *before* explicitly generating its internal reasoning (thought process). While our formatting script attempts to reorder this to `<think>...</think>` before the tool call for SFT, the original data's sequence might reflect a less effective agent strategy that contributes to "insufficiently robust tool-usage policies."

## 3. Recommendations & Next Steps

*   **Immediate (Phase 2):** Proceed with SFT on the **full 287 conversations** to establish a strong baseline. The training pipeline has been validated, and the data is now fully aligned.
*   **Long-Term (Phase 2 & 3):**
    *   **Implement Dense Rewards:** Proceed with the planned dense reward shaping in `slime/envs/tau/wrappers.py` to provide richer feedback during RL, addressing the "sparse rewards" issue.
    *   **Refine Tool Usage:** Monitor tool call generation during SFT and RL closely to ensure the model learns to effectively reason and act.
