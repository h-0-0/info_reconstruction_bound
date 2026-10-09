# From Pixels to Privacy-Sensitive Features: Information-Theoretic reconstruction bounds for DP-SGD

Code for *From Pixels to Privacy-Sensitive Features: An Information-Theoretic Framework for Reconstruction Bounds*.

For a DP-SGD training schedule — sampling rates and noise multipliers $(q_t, \sigma_t)_{t=1}^T$ — the code computes a
**lower bound $\rho^\star$ on the expected reconstruction error of every attack**, at 95% confidence, in pixel space
and in LPIPS feature space. It combines

* the **leakage** $\gamma$ of the schedule (Thm. `thm:gc_canonical`; `src/privacy/gamma.py`), which depends only on
  $(q_t, \sigma_t)$, and
* a **finite-sample lower confidence bound on the source entropy** of orthogonal blocks of the representation
  (Prop. `prop:coverage`; the plug-in estimator of Goldfeld et al. with Monte-Carlo and McDiarmid margins,
  `src/bound/`), estimated on held-out data,

through rate–distortion (Cor. `cor:rp`; `src/bound/rho.py`). Two experiments use it:

* **the privacy spectrum** (Fig. 1b, `fig:spectrum_all`, `tab:spectrum*`): the bound and the test accuracy of a
  realistic DP-SGD run (LeNet-5 / WRN-16-4 / ResNet-18 with GroupNorm, augmentation multiplicity, EMA) at each target
  $\varepsilon$, on MNIST, CIFAR-10, CelebA and ImageNette;
* **tightness at fixed leakage** (`fig:tightness*`, `tab:attack_results`, `tab:tightness`): the bound against the
  prior-aware attack of Hayes et al. on DP-SGD and against a reconstruction-optimised codebook release, on MNIST and
  CelebA.

## Regenerate every figure and table (CPU, minutes)

The repository ships the outputs of every compute-heavy step (`results/`, 5.5 MB). The three notebooks turn them into
the paper's figures (`figures/`, under the names the paper includes) and tables (`tables/`).

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
for nb in paper_utility_curves paper_tightness paper_bound_details; do
    SOURCE_DATE_EPOCH=0 .venv/bin/python -m nbconvert --to notebook --execute --inplace notebooks/$nb.ipynb
done
git status figures tables
```

(`SOURCE_DATE_EPOCH=0` fixes the timestamps matplotlib writes into PDFs, so regenerated figures are byte-identical to
the committed ones; with the pinned `requirements.txt`.)

Next to each figure the notebooks print the numbers the paper quotes, and `CHECK` lines for its qualitative claims.
`paper_bound_details` downloads the LPIPS (AlexNet) weights on first use.

| paper | file | notebook | produced by |
|---|---|---|---|
| Fig. 1b (`fig:spectrum_headline`) | `figures/utility_celeba_paper.pdf` | `paper_utility_curves` | bounds + fleet |
| `fig:spectrum_all` | `figures/utility_{mse,lpips}.pdf` | `paper_utility_curves` | bounds + fleet |
| `tab:spectrum`, `tab:spectrum_full` | `tables/spectrum{,_full}.tex` | `paper_utility_curves` | bounds + fleet |
| `fig:tightness`, `fig:tightness_mnist` | `figures/tightness_{celeba,mnist}.pdf` | `paper_tightness` | bounds + tightness stage |
| `tab:attack_results` | `tables/attack_results.tex` | `paper_tightness` | `scripts/tightness_spm.py` |
| `tab:tightness` | `tables/tightness.tex` | `paper_tightness` | `scripts/tightness_{codebook,subbit}.py` |
| `fig:sandwich_full` | `figures/tightness_pertarget.pdf` | `paper_tightness` | `scripts/tightness_spm.py` |
| `fig:exemplars`, `fig:exemplars_celeba` | `figures/tightness_exemplars_*.pdf` | `paper_tightness` | `scripts/tightness_spm.py --exemplar-mode` |
| dataset table, `tab:trsigma` | `tables/{datasets,trsigma}.tex` | `paper_bound_details` | `scripts/sigma_trace_ci.py` |
| grid, margins, Hadamard orders, accountant accuracy (`app:exp_entropy`) | printed | `paper_bound_details` | bounds |

## Reproduce from scratch (GPU cluster)

One script schedules every stage with explicit dependencies, on SLURM (`LAUNCHER=slurm`) or sequentially on one
machine (`LAUNCHER=local`). Set partitions and datasets in `reproduce/config.sh`, then

```bash
bash reproduce/run_all.sh                 # everything, in dependency order
bash reproduce/run_all.sh bounds          # or one stage: setup | bounds | tightness | dpsgd | notebooks
```

| stage | what | output |
|---|---|---|
| `bounds` | the entropy confidence bound per dataset × {pixel, LPIPS}: one GPU job per grid pair (36–60), then assembly | `results/bounds/{pixel,lpips}_{ds}/entropy_bound.json` |
| `tightness` | codebook and sub-bit releases, the SPM attack (+ exemplars, image inversion), tr Σ confidence bounds | `results/tightness/`, `results/tightness_spm/`, `results/sigma_trace_ci.json` |
| `dpsgd` | non-private baselines (lr search), the fleet table ($\varepsilon \mapsto \sigma$), 132 DP-SGD runs of 100 epochs | `results/utility/` |
| `notebooks` | figures and tables | `figures/`, `tables/` |

**Compute.** As a rough guide, reproducing everything on a single 11 GB GPU (e.g. an RTX 2080 Ti), one job at a
time, takes about 1,000 GPU hours (six weeks). Approximate hours by dataset:

| dataset | bound (pixel + LPIPS) | tightness | DP-SGD training | total |
|---|---|---|---|---|
| MNIST | 90 | 2 | 10 | 100 |
| CIFAR-10 | 70 | – | 250 | 320 |
| CelebA | 90 | 4 | 400 | 500 |
| ImageNette | 10 | – | 90 | 100 |

These are estimates from our own runs, which used a mix of GPUs. On a cluster the bound's grid pairs and the
training runs are independent jobs, so the wall-clock time is much shorter. The notebooks take minutes on a CPU.

**Data.** MNIST and CIFAR-10 download through torchvision, and CelebA from a HuggingFace mirror, on first use.
ImageNette is placed by hand: extract
[imagenette2-320.tgz](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz) to
`data/imagenette/imagenette2-320/`. The attack splits are fixed by `results/attack/splits/` (seed 42).

**Determinism.** The CPU steps, including every figure and table made from the shipped results, reproduce exactly.
The GPU steps do not: random-number streams and the order of parallel sums differ between GPU models (and some
between runs), so re-runs aren't bit for bit identical. 

Every step can also be run by hand from the repository root, e.g.
`.venv/bin/python scripts/sigma_trace_ci.py --datasets mnist`.
