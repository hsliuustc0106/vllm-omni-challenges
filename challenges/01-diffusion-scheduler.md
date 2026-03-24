# Challenge 01: Diffusion Scheduler for vllm-omni

## Summary

Design a fine-grained scheduler and memory-management mechanism for multimodal diffusion inference in `vllm-omni`, targeting mixed workloads with heterogeneous request lengths, resolutions, and frame counts.

The challenge is to improve SLO attainment under dynamic load while reducing end-to-end tail latency.

## Technical Background

Large-model serving has evolved from pure LLM workloads toward multimodal and omni-modal workloads. In practical internet serving scenarios, diffusion transformer workloads such as `Qwen-Image` and `WAN 2.2` introduce substantial inference pressure because requests vary significantly in:

- spatial resolution
- number of generated frames
- denoising step count
- execution duration
- memory footprint

Under a traditional FIFO scheduler, long-running generation requests can block short urgent requests. This causes queue buildup, severe SLO violations, and unstable latency under bursty traffic.

This challenge focuses on a scheduler that is aware of dynamic load, request heterogeneity, and memory pressure, with support for fine-grained preemption or resource arbitration across diffusion-generation stages.

## Problem Statement

Given a mixed request queue containing both long and short diffusion-generation requests, design a serving mechanism for `vllm-omni` that jointly optimizes:

- scheduling policy
- memory allocation and reclamation
- request admission and prioritization
- interruption, preemption, or timeslicing strategy where applicable

The solution should leverage the iterative generation nature of diffusion models, rather than treating each request as a single non-preemptible atomic job.

## Current Baseline Limitations

### 1. Coarse-grained static resource allocation

Current multimodal generation inference is often implemented as an atomic, non-preemptive workflow. Resources are effectively reserved once a task starts, and the inference process is treated as difficult to interrupt safely.

Impact:

- long jobs monopolize accelerator resources
- transient short jobs cannot be served quickly
- average latency and tail latency both degrade

### 2. Shortest-remaining-time style heuristics are insufficient

Even if a shortest-remaining-time-first style policy is introduced, lower-level runtime and memory constraints may still prevent real responsiveness in dynamic serving environments.

Impact:

- the scheduler cannot react quickly enough to bursty short requests
- active long jobs may pin NPU memory and execution slots
- queueing delay and latency jitter remain severe

## Technical Challenge

Jointly design an efficient memory-management mechanism and scheduling strategy that improves SLO attainment and reduces average end-to-end latency under heterogeneous workloads.

Key directions:

- Fine-grained dynamic resource management
  Use the iterative nature of DiT inference and build a resource model based on denoising progress, resolution, parallel strategy, and memory usage.
- Dynamic mixed-workload scheduling
  For dynamically arriving long and short requests, explore intelligent priority control, preemption, timeslicing, or staged execution strategies.

## Expected Scope

Solutions may include one or more of the following:

- request classification based on estimated runtime or memory footprint
- scheduler policies aware of denoising-step progress
- memory-pool design for diffusion request contexts
- context eviction, swap, checkpoint, or recomputation mechanisms
- admission control under overload
- SLO-aware priority boosting for short requests
- batch shaping or request regrouping for compatible jobs
- stage-level or iteration-level preemption

## Target Models and Workload

Validate the solution using the designated industry-standard models and request load:

- diffusion-capable multimodal models such as `Qwen-Image` and `WAN 2.2`
- request workloads derived from `vllm-omni/benchmark`
- the diffusion benchmark dashboard under `benchmarks/diffusion/performance_dashboard`

Validation should cover both model behavior and workload behavior.

## Constraints

Feel free to work on this challenging problem on any hardware/software version/time since `vllm-omni` supports various hardware. If you want to participate in NPU track,the challenge should be run under explicitly declared constraints. Unless a competition organizer overrides them, use the following categories as mandatory requirements:

### Hardware

- restrict the runtime hardware to `8 x Ascend 910B 64G`
- do not allow submissions to use more or fewer accelerator cards for official results
- treat `64 GB` device memory per card as the fixed hardware budget
- declare host CPU, system memory, and storage assumptions if they affect reproducibility
- do not compare results across mismatched hardware without normalization or separate reporting

### vllm-omni Version

- restrict the evaluated software version to the latest `vllm-omni` release candidate: `v0.18.0rc1`
- do not use release candidates, nightly builds, or unreleased commits for official results
- require participants to report the exact code version and patch set used in experiments
- if model runners or benchmark harnesses are modified, those changes must be disclosed

### Challenge Open Duration

- set the submission deadline to `2026-03-31 23:59:59 Asia/Shanghai`
- freeze the baseline version and benchmark rules for the full open period, or announce changes explicitly

For this repository draft, the submission deadline is fixed, while the start date and final review window should still be filled in by the organizer when the challenge is officially released. 


## Technical Targets

The solution should aim to achieve the following:

1. SLO attainment for multimodal diffusion inference requests greater than `99%`
2. More than `50%` reduction in `P95` end-to-end latency

These targets should be measured against a clearly stated baseline.

## Evaluation Guidance

Proposed solutions should be evaluated on at least these axes:

### Serving Metrics

- SLO attainment rate
- average end-to-end latency
- `P95` and `P99` latency
- queueing delay
- throughput under sustained load

### Resource Metrics

- NPU memory utilization
- memory fragmentation or pool efficiency
- preemption overhead
- recomputation overhead
- accelerator occupancy

### Robustness

- behavior under bursty traffic
- fairness between long and short requests
- stability under mixed resolutions and frame counts
- degradation mode under overload

## Deliverables to the vllm-omni repo

Participants should provide:

1. A design document describing the scheduler and memory-management strategy
2. Implementation details inside or around `vllm-omni`, provide us with a commit to the targeted vllm-omni release
3. Benchmark configuration and workload assumptions
4. Baseline-versus-proposed comparison results
5. Analysis of tradeoffs, failure modes, and overheads

## Suggested Solution Questions

Authors may use the following as framing questions:

1. At what granularity can diffusion inference be paused safely?
2. Which request features best predict remaining time and memory demand?
3. How should memory be partitioned between active and paused requests?
4. When should a long request yield to a short request?
5. How much preemption overhead is acceptable before gains disappear?
6. How should SLO policies adapt under overload?

## References

1. Xia, Yifei, et al. "TridentServe: A Stage-level Serving System for Diffusion Pipelines." `arXiv:2510.02838`, 2025.
2. `vllm-project/vllm-omni`: <https://github.com/vllm-project/vllm-omni>
3. Diffusion performance dashboard: <https://github.com/vllm-project/vllm-omni/tree/main/benchmarks/diffusion/performance_dashboard>

## Notes

This challenge brief is derived from the provided problem statement describing a diffusion-scheduler problem for `vllm-omni`, with emphasis on:

- dynamic SLO-aware scheduling
- fine-grained diffusion resource management
- long/short request coexistence
- NPU memory-pool coordination
