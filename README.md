# vllm-omni Challenges

This repository collects challenge problem statements for `vllm-omni`.

The intent is to turn high-level competition or research topics into reusable challenge briefs with:

- a clear technical background
- a concrete problem statement
- explicit constraints
- measurable targets
- expected deliverables
- references to baseline systems and upstream projects

## Structure

- `challenges/`: individual challenge briefs
- `templates/`: reusable authoring template for new problems

## Current Challenges

1. `01-diffusion-scheduler.md`
   Diffusion inference scheduling and memory management for mixed long/short requests under SLO constraints.

## Authoring Notes

Each challenge should be self-contained and answer:

1. What system problem is being solved?
2. Why does the current baseline fall short?
3. What exact technical work is expected?
4. What constraints apply to hardware, `vllm-omni` version, and challenge duration?
5. How should solutions be evaluated?
6. What artifacts should participants submit?

## Standard Constraints

Each challenge brief should include a `Constraints` section. At minimum, define:

- hardware constraints
- required `vllm-omni` version, branch, or commit range
- challenge open duration, including start date and end date
