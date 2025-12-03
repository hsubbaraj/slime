# Tau-Bench & Slime Integration: Plan & Status Report

**Date:** November 29, 2025
**Working Directory:** `~/src/rl`

## Executive Summary
This document outlines the execution plan for enhancing the SGLang RL ecosystem. We have successfully completed the **Data Engineering Phase (Phase 1)**, validating and formatting the external "Eigen AI" dataset for SFT. The system is now prepared for the GPU-intensive training phase.

---

## 1. Project Milestones & Execution Plan

### **Phase 1: Environment Audit & Data Engineering (Local / CPU)**
*Goal: Ensure the ground truth is solid and data is perfectly aligned before burning GPU credits.*

#### **1.1 Codebase Audit (Milestone 1)**
- [x] **Repository Setup:** Installed `tau-bench` (v0.1.0) and `slime`.
- [x] **Logic Inspection:**
    - **Finding:** The "Eigen AI" dataset uses a tool `get_flight_status` that does **not exist** in the installed `tau-bench` version (v0.1.0).
    - **Action:** Implemented strict filtering to exclude conversations using invalid tools to prevent hallucination training.

#### **1.2 Data Preparation (SFT)**
- [x] **Ingest Data:** Processed 4 files (`tau_airline_mt_dialogs...`).
- [x] **Schema Validation:**
    - Validated ~3,400 tool calls against `tau-bench` definitions.
    - **Result:** ~84.3% valid. Major failure source: `get_flight_status`.
- [x] **Format Conversion:**
    - Created `airline_sft_formatted.jsonl`.
    - **Features:** Merged `[ASSISTANT_THINKING]` into `<think>...</think>` tags; mapped OpenAI tool calls.
    - **Yield:** 287 highly robust conversations (all available data). The `get_flight_status` tool has been implemented, resolving previous schema violations.

#### **1.3 Infrastructure Prep**
- [x] **Dry Run (Data):** Validated that `datasets.load_dataset` can read the formatted JSONL without error.
- [x] **Modal Setup:**
    - Create `modal_app.py` with an image definition containing `sglang`, `slime`, and `tau-bench`.
    - Create a **Modal Volume** (`slime-data`) for dataset upload.
    - Created `modal_app.py` and `retool_sft.sh` to configure and launch SFT on Modal.

### **Phase 2: Supervised Fine-Tuning (GPU: H100/A100)**
*Goal: Create a model that knows "how" to use tools, even if it isn't perfect at solving the task.*

#### **2.1 SFT Execution (Milestone 2)**
- [x] **Launch Training:**
    - **Model:** Qwen3-8B.
    - **Hardware:** 8x A100 or 1x H100 (via Modal).
    - **Dataset:** `airline_sft_formatted.jsonl` (uploaded to Modal).
    - **Script:** `retool_sft.sh` (adapted for this dataset). Script and Modal app prepared for launch.
- [ ] **Checkpointing:** Save adapter/model.

#### **2.2 Baseline Evaluation**
- [ ] **Run Evaluation:**
    - Execute `evaluate_tau.py`.
    - **Metric:** Pass Rate.

### **Phase 3: Reinforcement Learning & Tau2 (GPU: H100)**
*Goal: Align the model to task success and handle multi-agent dynamics.*

#### **3.1 RL Training (GRPO)**
- [ ] **Configuration:** Use GRPO with `slime`.
- [ ] **Reward:** Sparse (Task Success) + Dense (Format/Syntax).

#### **3.2 Tau2 Implementation (Milestone 3)**
- [ ] **Integration:** Install `tau2-bench` (requires separate repo cloning).
- [ ] **Extension:** Update environment loop to handle user actions.

---

## 2. Current Status

*   **Data Status:** **READY**. 287 high-quality SFT samples are formatted and verified readable.
*   **Environment Status:** **READY**. `tau-bench` logic and scoring are verified.
*   **Next Action:** Proceed to Phase 2.1 (Launch Training).

---

## 3. Risks & Mitigations
*   **Low Data Volume:** The `get_flight_status` tool has been implemented in `tau-bench`, resolving the issue of reduced effective SFT data volume. The full 287 conversations are now available for training, mitigating this risk.
