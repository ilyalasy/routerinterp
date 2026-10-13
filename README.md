# RouterInterp: Understanding Superposed Specialisation in Mixture of Experts Routing

[![Tests](https://github.com/ilyalasy/routerinterp/actions/workflows/tests.yml/badge.svg)](https://github.com/ilyalasy/routerinterp/actions/workflows/tests.yml)

This repo contains the codebase for our method of interpreting the router in MoEs using Sparse Autoencoder (SAE) features. Details are in the [paper](https://arxiv.org/abs/2610.11775).

## Setup

```bash
poetry install
cp .env.example .env
```

Fill in `OPENROUTER_API_KEY`, `WANDB_PROJECT`, and `ARTIFACTS_ROOT`.

Dev checks are [poethepoet](https://github.com/nat-n/poethepoet) tasks: `poe lint`, `poe format`, `poe typecheck`, `poe test`. `poe check-all` runs all of them.

## Pipeline

```
collect
 ├─ select ── explain ── evaluate
 │     │         │
 │     │         └─ cluster
 │     └─ feature-selection sweep
 ├─ probe
 └─ pile-domains
```

- **collect** ([collect.sh](scripts/slurm/collect.sh)). Run the model and its SAE over the corpus. Save tokens, SAE activations, the router's top-k experts, residual stream, and each expert's output. Every later step reads this cache.
- **select** ([select.sh](scripts/slurm/select.sh)). Every SAE feature is scored against every expert, by how predictive this feature is of activating this expert.
- **explain** ([explain_features.sh](scripts/slurm/explain_features.sh), [explain_experts.sh](scripts/slurm/explain_experts.sh)). Read examples from the corpus, split into short windows and feed to the LLM to generate a text description of the target (can be either an expert or a feature). Similar to [delphi](https://github.com/EleutherAI/delphi).
- **evaluate** ([eval_experts.sh](scripts/slurm/eval_experts.sh)). Score those descriptions on held out windows, using scorer LLM. Similar to [delphi](https://github.com/EleutherAI/delphi).
- **cluster** ([feature_clustering.sh](scripts/slurm/feature_clustering.sh)). Embed the feature descriptions and count how many clusters each expert's features fall into.
- **probe** ([probes.sh](scripts/slurm/probes.sh)). Fit a linear probe that predicts expert activation from SAE features, neurons, PCA components, or n-grams.
- **feature-selection sweep** ([feature_selection_sweep.sh](scripts/slurm/feature_selection_sweep.sh)). Repeat select across scoring methods and across number of features *n*, then measure how well each set predicts which expert fires.
- **pile-domains** ([pile_domains.sh](scripts/slurm/pile_domains.sh)). Measure which Pile domains each expert prefers. Similar to expert specialisation [Mixtral experiment](https://arxiv.org/abs/2401.04088).

## Code

All CLI entrypoints are defined in [cli.py](src/routerinterp/cli.py). Here is the list of all most important files/folders:

| | |
| --- | --- |
| [backends.py](src/routerinterp/backends.py) | One interface for both models and both SAE libraries. Models: OLMoE (`olmoe`) and GPT-OSS-20B (`gpt_oss`). SAEs: [sparsify](https://github.com/EleutherAI/sparsify) (a local directory or a W&B artifact) and [dictionary_learning](https://github.com/saprmarks/dictionary_learning) (Hugging Face or a local directory). |
| [collect.py](src/routerinterp/collect.py) | Writes the activation cache. |
| [select.py](src/routerinterp/select.py) | Calculate score per feature x expert pair, representing how a particular feature is predictive of an expert.  |
| [explain.py](src/routerinterp/explain/explain.py) | Explanation methods. The list is [`EXPLAINERS`](src/routerinterp/explain/explain.py#L395). |
| [evaluate.py](src/routerinterp/explain/evaluate.py) | Evaluate explanations using scorer LLM. |
| [experiments/](src/routerinterp/experiments/) | Reads finished pipeline outputs and computes metrics for three paper results: SSH vs DSH experiment on feature clustering, routing probes, the pile domain expert preference. |
| [plot.py](src/routerinterp/plot.py) | The paper's data figures: clusters per expert, the feature-selection sweep, Pile domains, explanation length, and explanation F1 by depth. |

## Methods

**Select** ([select.py](src/routerinterp/select.py), `--method`):

| method | description |
| --- | --- |
| `routing_margin` | Main RouterInterp feature selection method. The routing margin is the gap between an expert's logit and the threshold it must clear to enter the top-k, the k-th largest logit among the other experts. Features are scored by how much of their positive effect on that gap is specific to tokens routed to the expert: the mean positive attribution when it is selected, minus λ times the same mean when it is not. |
| `rho_useful` | How much higher a feature's average activation is when the expert is in the top-k than when it is not. |
| `cosine` | Cosine between the expert's router weight and the feature's decoder direction. |

**Explain** ([explain.py](src/routerinterp/explain/explain.py), `--method`):

| method | description |
| --- | --- |
| `routerinterp` | Generates description per expert. The prompt shows windows where a selected feature fires and the expert is routed, and windows where the feature fires and the expert is not. |
| `feature_autointerp` | Generates description per SAE feature, from the windows where that feature activates the most. |
| `expert_activation` | Generates description per expert, from windows where that expert is routed - the most naive baseline. |
| `esb` | [Expert Strikes Back](https://arxiv.org/abs/2604.02178) / Herbst method; the paper calls it Expert Impact AutoInterp. Generates description per expert. Windows are ranked by router weight times the size of the expert's output, and the prompt includes the tokens that output promotes (logit lens). |
| `unigram_autointerp` | An LLM description of the unigrams (tokens) the expert fires on most often. |
| `bigram_autointerp` | An LLM description of the bigrams (token pairs) the expert fires on most often. |
| `unigram_lookup` | The expert's 100 most common unigrams (tokens), stored as a table. |
| `bigram_lookup` | The expert's 100 most common bigrams (token pairs), with a fallback to its unigram table. |

## Outputs

### Artifacts

Each stage writes its own directory and a `manifest.json`: the hash of the resolved config, and the hashes of the parent stages. A rerun with the same config and the same parents returns immediately. `collect` also resumes shard by shard. `explain` skips a target whose prompt hash is already in `explanations.jsonl`. `evaluate` skips targets already in `eval_progress.jsonl`.

Paths in the configs are relative, such as `artifacts/olmoe/collect`. Set `ARTIFACTS_ROOT` and they are stored under that directory.

### Weights & Biases

When `WANDB_PROJECT` is set, a finished stage is logged as a run. Small parquet files become tables. The run directory is uploaded as one versioned artifact.

The paper runs are in <https://forge.coreweave.com/wandb/moe-interp/routerinterp>. A run is one explanation method and one scoring seed. The name is `explain-<method>-s<seed>` for the explanation job and `evaluate-<method>-s<seed>` for a scoring job. GPT-OSS numbers in the paper are the mean of seeds 0, 1, and 2. Search the run list by that name, or filter on the `method` tag. The group is the model and the layers (`gpt-oss-20b/L3-7-11-15-19`, `OLMoE-1B-7B-0125/L2-6-10-14`). A paper layer L is the hookpoint `layers.(L-1)`, because `layers.N` is the residual after layer N and the input to router N+1; the groups `gpt-oss-20b/L3-7-11-15-19` and `OLMoE-1B-7B-0125/L2-6-10-14` are those hookpoint indices. RouterInterp with 45 features, seed 0, is three runs, split by layer.

The OLMoE SAE (k = 32, expansion 16, layers 2/6/10/14, trained on OLMoE-mix) is [layers.2-layers.6-layers.10-layers.14-32-16](https://forge.coreweave.com/wandb/moe-interp/routerinterp/artifacts/sparse_autoencoder/layers.2-layers.6-layers.10-layers.14-32-16/v0). [configs/olmoe.yaml](configs/olmoe.yaml) loads `:latest` of that artifact.

## Reproduction

The jobs in [scripts/slurm/](scripts/slurm/) ran on the [Isambard](https://docs.isambard.ac.uk/) SLURM cluster. Whenever the GPU was needed we mostly used 1 GH200.

### Configs

[configs/](configs/) stores configs per model: [olmoe.yaml](configs/olmoe.yaml), [gpt_oss_s64.yaml](configs/gpt_oss_s64.yaml), [gpt_oss_s128.yaml](configs/gpt_oss_s128.yaml). Each file is the whole pipeline for that model (`collect`, `select`, `explain`, `evaluate`, `probes`). OLMoE also has `sae_training`.

### Scripts

| script | job |
| --- | --- |
| [run.sbatch](scripts/slurm/run.sbatch) | Load `.env` and run one `routerinterp` command. |
| [collect.sh](scripts/slurm/collect.sh) | Cache activations for each model. |
| [select.sh](scripts/slurm/select.sh) | Generate scores `features x experts` for each model. |
| [explain_features.sh](scripts/slurm/explain_features.sh) | Generate feature explanations for each model. |
| [explain_experts.sh](scripts/slurm/explain_experts.sh) | Generate expert explanations for every method per model. |
| [eval_experts.sh](scripts/slurm/eval_experts.sh) | Score those expert explanations. |
| [probes.sh](scripts/slurm/probes.sh) | SAE, neuron, PCA, and n-gram routing probes. |
| [feature_selection_sweep.sh](scripts/slurm/feature_selection_sweep.sh) | Select and probe across different number of features *n*. |
| [feature_clustering.sh](scripts/slurm/feature_clustering.sh) | Main SSH vs DSH evidence. Cluster most predictive features of experts. |
| [explanation_length.sh](scripts/slurm/explanation_length.sh) | Routerinterp's explanations scoring and length versus feature-set size. |
| [pile_domains.sh](scripts/slurm/pile_domains.sh) | Pile-domain routing preference. |
| [train_sae.sh](scripts/slurm/train_sae.sh) | Retrain the OLMoE SAE. |

## Citation

```bibtex
@inproceedings{lasy2026routerinterp,
  title     = {RouterInterp: Understanding Superposed Specialisation in Mixture of Experts Routing},
  author    = {Ilya Lasy and Nora Yinuo Cai and Kola Ayonrinde},
  booktitle = {Forty-third International Conference on Machine Learning},
  year      = {2026},
  url       = {https://openreview.net/forum?id=rDNCWfRd69}
}
```
