# Tau-1 Audit Report: Reasons for Suboptimal Model Performance

**Date:** November 29, 2025
**Updated:** December 4, 2025

## 1. Executive Summary
This report details findings regarding the suboptimal performance of SGLang-trained Qwen models in the Tau1 (Airline) environment and the technical challenges encountered during SFT setup.

**Initial Issues (November 2025):**
*   Tool mismatch between SFT data and tau-bench environment (resolved by implementing `get_flight_status`)
*   Sparse reward structure requiring dense reward shaping for RL
*   Potential suboptimal trajectory format in training data

**Critical Discovery (December 2025):**
*   **83.6% of tau-bench airline conversations exceed 8,192 tokens** - far longer than typical SFT datasets
*   Median conversation length: 11,834 tokens (P95: 23,177 tokens, Max: 36,524 tokens)
*   Standard training configurations encounter immediate CUDA Out of Memory errors

**Solution Implemented:**
*   Context Parallel (CP) training with sequence splitting across GPUs using Ring Attention
*   Configuration: CP=2, batch size 1, 12K tokens per GPU (24K total capacity)
*   Retains 95% of training data (268 of 287 conversations)
*   Trade-off: ~15-25% slower training but handles realistic conversation complexity

**Current Status:**
*   Training infrastructure configured and ready for validation testing
*   Staged testing workflow established (quick test → full test → production)
*   Hardware requirements defined (2 GPUs minimum for testing, 8+ recommended for production)

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

### 2.4 Sequence Length & Memory Challenges (December 2025 Update)
*   **Critical Discovery - Extremely Long Sequences:** Token length analysis of the 287-conversation SFT dataset revealed that tau-bench airline conversations are significantly longer than typical SFT datasets:
    *   **83.6% of conversations exceed 8,192 tokens** (standard limit for many training setups)
    *   **Median length: 11,834 tokens**
    *   **P95: 23,177 tokens**
    *   **P99: 29,516 tokens**
    *   **Maximum: 36,524 tokens**

*   **Impact:** This finding invalidated the initial approach of filtering conversations to fit standard memory constraints, as it would have eliminated the majority of training data.

*   **Technical Challenges Encountered:**
    1. **Initial OOM Error:** First training attempt with `--max-tokens-per-gpu 8192` immediately failed with CUDA Out of Memory when processing a 12,557-token conversation.
    2. **Naive Filtering Rejected:** Filtering to 8K tokens would have removed 240 of 287 conversations (83.6%), destroying the dataset.
    3. **Context Parallel Configuration Error:** Second attempt with `--context-parallel-size 2` and `--global-batch-size 2` on 2 GPUs failed due to incompatible parallelism settings (CP=2 uses both GPUs for sequence splitting, leaving no GPUs for data parallelism).

*   **Root Cause Analysis:**
    *   Tau-bench airline conversations involve complex multi-turn interactions with extensive tool usage, reasoning traces, and error handling
    *   Each conversation includes multiple rounds of: user request → agent thinking → tool calls → tool results → agent response
    *   The `<think>` tags containing agent reasoning significantly increase token counts
    *   This is not a data quality issue but rather reflects the realistic complexity of production customer service scenarios

### 2.5 Solution Implemented: Context Parallel Training
*   **Configuration:**
    *   Enabled Context Parallel (CP) with `--context-parallel-size 2` to split sequences across GPUs using Ring Attention
    *   Adjusted `--max-tokens-per-gpu 12288` (12K per GPU = 24K total capacity)
    *   Filtered only conversations exceeding 24,576 tokens (~5% of dataset)

*   **Trade-offs:**
    *   ✅ Retains 95% of training data (268 of 287 conversations)
    *   ✅ Handles realistic conversation lengths up to P95
    *   ⚠️ Training is ~15-25% slower due to Ring Attention communication overhead

*   **Hardware Requirements:**
    *   **Minimum for testing:** 2x H100 GPUs with CP=2, batch size 1
    *   **Recommended for production:** 8+ GPUs to enable both CP and DP (e.g., CP=2 × DP=4)

### 2.6 Solution Optimization: Batch Size Tuning
*   **Cost vs. Throughput Analysis:** An optimization script (`optimize_cost.py`) was developed to evaluate the trade-offs between GPU count, Context Parallel size, and dataset coverage.
*   **Selected Configuration:** **4 GPUs with CP=2 and DP=2**.
    *   **Why:** This setup retains 95.1% of the data (up to 24k tokens) and offers the "Best Value" in terms of estimated training cost ($0.78 per run vs $0.92 for 100% coverage on 8 GPUs).
*   **Batch Size Strategy:**
    *   **Micro Batch Size:** Set to `1` (per DP group) to strictly respect memory constraints with 24k token sequences.
    *   **Global Batch Size:** Set to `4`. With DP=2, this implies `Gradient Accumulation Steps = 2`.
    *   **Impact:** This configuration improves throughput by reducing optimizer step frequency while keeping peak memory usage within safe limits.

## 3. Recommendations & Next Steps

### 3.1 Immediate Actions (Phase 2 - SFT Launch)
*   **Training Configured:** The SFT job `run_sft_job` has been updated to use:
    *   **Hardware:** 4x H100 GPUs
    *   **Parallelism:** CP=2, DP=2
    *   **Batching:** Micro=1, Global=4
    *   **Epochs:** 3
*   **Execution:** Launch `modal run slime/examples/tau-bench/modal_app.py::run_sft_job` to begin the 3-epoch SFT run. Real-time logging is enabled by default.

### 3.2 Short-Term (Phase 2 Optimization)
*   **Monitor Convergence:** Watch the loss curve closely. With a Global Batch Size of 4 and a small dataset (268 samples), updates will happen frequently (~67 steps per epoch). Ensure the loss decreases stably and doesn't oscillate wildly.
*   **Verify Evaluation:** After training, run `evaluate_tau.py` to confirm the model has learned the tool-use syntax and can successfully interact with the airline environment.

### 3.3 Long-Term (Phase 3 - RL)
*   **Dense Reward Shaping:** Implement dense rewards in `slime/envs/tau/wrappers.py` to address sparse reward challenge
*   **RL with Context Parallel:** Extend CP configuration to RL training phase (both rollout and training)
*   **Sequence Length Analysis for RL:** Monitor if RL generates even longer sequences during exploration

### 3.4 Infrastructure Considerations
*   **Memory Profiling:** Continue monitoring GPU memory usage to optimize `--max-tokens-per-gpu` setting
*   **Alternative Approaches (if needed):**
    *   Consider gradient checkpointing for additional memory savings
    *   Explore FlashAttention optimizations for CP mode
    *   Evaluate sequence packing strategies that respect CP boundaries

## 4. Technical Artifacts & References

**Modal Functions:**
*   `examples/tau-bench/modal_app.py::filter_long_conversations` - Data filtering with tokenizer
*   `examples/tau-bench/modal_app.py::create_tiny_test_data` - Test dataset creation
*   `examples/tau-bench/modal_app.py::run_quick_test` - Fast validation test
*   `examples/tau-bench/modal_app.py::analyze_data_lengths` - Token length analysis
*   `examples/tau-bench/modal_app.py::run_sft_job` - Main SFT training job

**Configuration Files:**
*   `examples/tau-bench/retool_sft.sh` - Training script with CP=2, DP=2 configuration
*   `scripts/models/qwen3-4B-Instruct-2507.sh` - Model hyperparameters

**Key Configuration Values:**
*   **GPUs:** 4 (H100)
*   **Context Parallel Size:** 2
*   **Data Parallel Size:** 2
*   **Max Tokens Per GPU:** 12,288 (24,576 total)
*   **Micro Batch Size:** 1
*   **Global Batch Size:** 4
*   **Training Data:** 268 conversations (95.1% of original dataset)
*   **Filtered Out:** 19 conversations (4.9%, exceeding 24K tokens)
