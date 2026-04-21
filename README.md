# HSTU Ranking on KuaiRand

This fork takes the **industrial ranking (DLRM-v3) code** from Meta's
[generative-recommenders](https://github.com/meta-recsys/generative-recommenders)
release of HSTU and uses it as a single-GPU baseline on the public
[KuaiRand](https://kuairand.com/) dataset, with the goal of reproducing recent
sequential-ranking papers.

The original repository is heavily oriented toward Meta's internal stack and
trillion-parameter setups; this fork keeps the upstream model intact while
adding the small set of dataset, config, and environment changes needed to make
end-to-end training run on a single GPU outside Meta.

## Prerequisites

Tested on Ubuntu 22.04, CUDA 12.4, Python 3.10, single H200.

### 1. Python dependencies

```bash
pip3 install -r requirements.txt
# or, manually:
# pip3 install gin-config pandas fbgemm_gpu torchrec tensorboard
```

### 2. Triton + torch.inductor compatibility

The HSTU triton kernels use TMA APIs added in Triton 3.3+, but `torch==2.6.0`
pins `triton==3.2.0`. Two one-time adjustments are needed:

```bash
# (a) Upgrade Triton to a release that has tl.make_tensor_descriptor / triton.set_allocator
pip install --user 'triton==3.4.0'

# (b) Patch torch's inductor hints.py so its AttrsDescriptor lookup falls back gracefully
# (Triton 3.4 removed AttrsDescriptor; we never invoke torch.compile in this path,
# so the namedtuple shim is fine.)
python3 - <<'PY'
import pathlib
p = pathlib.Path(__import__('torch').__file__).with_name('_inductor')/'runtime'/'hints.py'
src = p.read_text()
old = '    except ImportError:\n        from triton.compiler.compiler import AttrsDescriptor\n\n        def AttrsDescriptorWrapper(\n            divisible_by_16=None,\n            equal_to_1=None,\n        ):\n            # Prepare the arguments for AttrsDescriptor\n            kwargs = {\n                "divisible_by_16": divisible_by_16,\n                "equal_to_1": equal_to_1,\n            }\n\n            # Instantiate AttrsDescriptor with the prepared arguments\n            return AttrsDescriptor(**kwargs)\n\nelse:'
new = '    except ImportError:\n        try:\n            from triton.compiler.compiler import AttrsDescriptor\n\n            def AttrsDescriptorWrapper(\n                divisible_by_16=None,\n                equal_to_1=None,\n            ):\n                kwargs = {"divisible_by_16": divisible_by_16, "equal_to_1": equal_to_1}\n                return AttrsDescriptor(**kwargs)\n\n        except ImportError:\n            AttrsDescriptorWrapper = collections.namedtuple(\n                "AttrsDescriptor", ["divisible_by_16", "equal_to_1"], defaults=[(), ()],\n            )\n\nelse:'
if old in src:
    p.write_text(src.replace(old, new))
    print('patched', p)
else:
    print('no patch needed')
PY
```

## Data preparation

KuaiRand-1K (~1k users, ~1 GB tar.gz, ~440 MB processed CSV):

```bash
mkdir -p data/
python3 generative_recommenders/dlrm_v3/preprocess_public_data.py --dataset kuairand-1k
```

KuaiRand-27K (~27k users, ~10 GB tar.gz, ~12 GB processed CSV):

```bash
python3 generative_recommenders/dlrm_v3/preprocess_public_data.py --dataset kuairand-27k
```

Both produce `data/KuaiRand-{1K,27K}/data/processed_seqs.csv`. Update
`make_train_test_dataloaders.new_path_prefix` in the matching gin file
(`generative_recommenders/dlrm_v3/train/gin/kuairand_{1k,27k}.gin`) if your
data lives outside the repo root.

## Training (single GPU, single-task `is_click`)

```bash
# 1k variant — quick smoke test (5 epochs, ~5 min on a single H200)
CUDA_VISIBLE_DEVICES=0 LOCAL_WORLD_SIZE=1 WORLD_SIZE=1 \
  python3 generative_recommenders/dlrm_v3/train/train_ranker.py \
  --dataset kuairand-1k --mode train-eval

# 27k variant — bigger and slower (5 epochs, ~1–2 h on a single H200, batch_size=64)
CUDA_VISIBLE_DEVICES=0 LOCAL_WORLD_SIZE=1 WORLD_SIZE=1 \
  python3 generative_recommenders/dlrm_v3/train/train_ranker.py \
  --dataset kuairand-27k --mode train-eval
```

Per-step `train` / `eval` metrics (NE, Accuracy, GAUC for `is_click`) are
printed to stdout. Tensorboard files land at `/tmp/tensorboard_log_path*.log`.

## What this fork changes vs. upstream

- **All KuaiRand features used.** `configs.py` enables 25 contextual features
  (`user_id`, `user_active_degree`, range buckets, `is_video_author`,
  `onehot_feat0..17`) plus `duration_ms` as a per-item sequence feature.
  `kuairand.py` cleans known data-quality issues (`is_live_streamer` int8
  overflow, NaN in float `onehot_feat*` columns) at dataset construction time.
- **Single task `is_click`** instead of the upstream 8-task multitask head.
  Most KuaiRand actions (`is_follow`, `is_forward`, `is_hate`, ...) have
  positive rates well below 1% on the candidate split, which made the
  multitask losses numerically unstable. To restore multitask, expand
  `multitask_configs` and `action_weights` in `configs.py`.
- **Bounded eval loop.** `train_eval_loop` only breaks the inner eval loop
  when `num_eval_batches` is set; otherwise it cycles the eval iterator
  forever after the first train step. The shipped gin files set this
  (`= 15` for 1k, `= 30` for 27k) and bump `eval_frequency` so eval doesn't
  fire after every train step.
- **Profiler off by default** (`output_trace=False`). The upstream profiler
  writes traces to a Meta-internal `manifold://` path that does not exist
  outside.
- **`triton.set_allocator` hasattr-guard** and a `try/except` around the
  optional `hammer.v2` `tlx_bw` import, so the code loads on stock PyPI
  Triton without Meta's internal extensions.

## License

Apache 2.0, inherited from the upstream
[meta-recsys/generative-recommenders](https://github.com/meta-recsys/generative-recommenders)
project.
