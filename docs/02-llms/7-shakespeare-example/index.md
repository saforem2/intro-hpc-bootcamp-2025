# [`wordplay` 🎮 💬](https://github.com/saforem2/wordplay): Shakespeare
✍️
Sam Foreman
2025-07-22

<link rel="preconnect" href="https://fonts.googleapis.com">

- [Install / Setup](#install--setup)
- [Post Install](#post-install)
- [Build Trainer](#build-trainer)
  - [Build `Trainer` object](#build-trainer-object)
- [Prompt (**prior** to training)](#prompt-prior-to-training)
- [Train Model](#train-model)
- [(partial) Training:](#partial-training)
- [Resume Training…](#resume-training)
- [Evaluate Model](#evaluate-model)

We will be using the [Shakespeare
dataset](https://github.com/saforem2/wordplay/blob/main/data/shakespeare/readme.md)
to train a (~ small) 10M param LLM *from scratch*.

<div>

<div align="center" style="text-align:center;">

<img src="https://github.com/saforem2/wordplay/blob/main/assets/shakespeare.jpeg?raw=true" width="45%" align="center" /><br>

Image generated from
[stabilityai/stable-diffusion](https://huggingface.co/spaces/stabilityai/stable-diffusion)
on [🤗 Spaces](https://huggingface.co/spaces).<br>

</div>

<details closed>

<summary>

Prompt Details
</summary>

<ul>

<li>

Prompt:
</li>

<t><q> Shakespeare himself, dressed in full Shakespearean garb, writing
code at a modern workstation with multiple monitors, hacking away
profusely, backlit, high quality for publication </q></t>

<li>

Negative Prompt:
</li>

<t><q> low quality, 3d, photorealistic, ugly </q></t>
</ul>

</details>

</div>

## Install / Setup

<div class="alert alert-block alert-warning">

<b>Warning!</b><br>

**IF YOU ARE EXECUTING ON GOOGLE COLAB**:

You will need to restart your runtime (`Runtime` $\rightarrow\,$
`Restart runtime`)  
*after* executing the following cell:

</div>

``` python
%%bash

python3 -c 'import wordplay; print(wordplay.__file__)' 2> '/dev/null'

if [[ $? -eq 0 ]]; then
    echo "Has wordplay installed. Nothing to do."
else
    echo "Does not have wordplay installed. Installing..."
    git clone 'https://github.com/saforem2/wordplay'
    python3 wordplay/data/shakespeare_char/prepare.py
    python3 wordplay/data/shakespeare/prepare.py
    python3 -m pip install deepspeed
    python3 -m pip install -e wordplay
fi
```

    /Users/samforeman/projects/saforem2/wordplay/src/wordplay/__init__.py
    Has wordplay installed. Nothing to do.

## Post Install

If installed correctly, you should be able to:

``` python
>>> import wordplay
>>> wordplay.__file__
'/path/to/wordplay/src/wordplay/__init__.py'
```

``` python
%load_ext autoreload
%autoreload 2
import os
import sys
import ezpz

os.environ['COLORTERM'] = 'truecolor'
if sys.platform == 'darwin':
    # If running on MacOS:
    # os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['TORCH_DEVICE'] = 'cpu'
# -----------------------------------------------

logger = ezpz.get_logger()

import wordplay
logger.info(wordplay.__file__)
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload
    [2025-07-24 17:54:34,596325][I][ipykernel_80295/2338663768:17:ezpz.log] /Users/samforeman/projects/saforem2/wordplay/src/wordplay/__init__.py

## Build Trainer

Explicitly, we:

1.  `setup_torch(...)`
2.  Build `cfg: DictConfig = get_config(...)`
3.  Instnatiate `config: ExperimentConfig = instantiate(cfg)`
4.  Build `trainer = Trainer(config)`

``` python
import os
import numpy as np
from ezpz import setup
from hydra.utils import instantiate
from wordplay.configs import get_config, PROJECT_ROOT
from wordplay.trainer import Trainer

HF_DATASETS_CACHE = PROJECT_ROOT.joinpath('.cache', 'huggingface')
HF_DATASETS_CACHE.mkdir(exist_ok=True, parents=True)

os.environ['HF_DATASETS_CACHE'] = HF_DATASETS_CACHE.as_posix()

BACKEND = 'DDP'

rank = setup(
    framework='pytorch',
    backend=BACKEND,
    seed=1234,
)

cfg = get_config(
    [
        'data=shakespeare',
        'model=shakespeare',
        'model.batch_size=1',
        'model.block_size=128',
        'optimizer=shakespeare',
        'train=shakespeare',
        f'train.backend={BACKEND}',
        'train.compile=false',
        'train.dtype=bfloat16',
        'train.max_iters=500',
        'train.log_interval=10',
        'train.eval_interval=50',
    ]
)
config = instantiate(cfg)
```

    [2025-07-24 17:54:34,636422][W][ezpz/dist:639] Caught TORCH_DEVICE=cpu from environment!
    [2025-07-24 17:54:34,637239][I][ezpz/dist:1303] Running on a single cpu, not initializing torch.distributed!
    [2025-07-24 17:54:34,647781][W][ezpz/dist:639] Caught TORCH_DEVICE=cpu from environment!
    [2025-07-24 17:54:34,648550][W][ezpz/dist:639] Caught TORCH_DEVICE=cpu from environment!
    [2025-07-24 17:54:34,670929][I][ezpz/dist:1377] Using device='cpu' with backend='gloo' + 'gloo' for distributed training.
    [2025-07-24 17:54:34,671784][I][ezpz/dist:1422] ['Sams-MacBook-Pro-2.local'][0/0] 
    [2025-07-24 17:54:34,723020][I][wordplay/configs:317] Loading train from /Users/samforeman/projects/saforem2/wordplay/data/shakespeare_char/train.bin
    [2025-07-24 17:54:34,735023][I][wordplay/configs:317] Loading val from /Users/samforeman/projects/saforem2/wordplay/data/shakespeare_char/val.bin
    [2025-07-24 17:54:34,765134][I][wordplay/configs:442] Tokens per iteration: 128
    [2025-07-24 17:54:34,766233][W][ezpz/dist:639] Caught TORCH_DEVICE=cpu from environment!
    [2025-07-24 17:54:34,767280][I][wordplay/configs:465] Using self.ptdtype=torch.bfloat16 on self.device_type='cpu'
    [2025-07-24 17:54:34,767969][I][wordplay/configs:471] Initializing a new model from scratch

### Build `Trainer` object

``` python
trainer = Trainer(config)
```

    [2025-07-24 17:54:34,805356][I][wordplay/trainer:235] Initializing a new model from scratch
    [2025-07-24 17:54:35,000428][I][wordplay/model:255] number of parameters: 10.65M
    [2025-07-24 17:54:35,002265][I][wordplay/trainer:252] Model size: num_params=10646784
    [2025-07-24 17:54:35,002973][I][wordplay/model:445] num decayed parameter tensors: 26, with 10,690,944 parameters
    [2025-07-24 17:54:35,003506][I][wordplay/model:449] num non-decayed parameter tensors: 13, with 4,992 parameters
    [2025-07-24 17:54:35,005495][I][wordplay/model:465] using fused AdamW: False
    [2025-07-24 17:54:35,006055][C][wordplay/trainer:308] "devid='cpu:0'"
    [2025-07-24 17:54:35,016686][I][wordplay/trainer:347] • self.model=GPT(
      (transformer): ModuleDict(
        (wte): Embedding(65, 384)
        (wpe): Embedding(128, 384)
        (drop): Dropout(p=0.2, inplace=False)
        (h): ModuleList(
          (0-5): 6 x Block(
            (ln_1): LayerNorm()
            (attn): CausalSelfAttention(
              (c_attn): Linear(in_features=384, out_features=1152, bias=False)
              (c_proj): Linear(in_features=384, out_features=384, bias=False)
              (attn_dropout): Dropout(p=0.2, inplace=False)
              (resid_dropout): Dropout(p=0.2, inplace=False)
            )
            (ln_2): LayerNorm()
            (mlp): MLP(
              (c_fc): Linear(in_features=384, out_features=1536, bias=False)
              (act_fn): GELU(approximate='none')
              (c_proj): Linear(in_features=1536, out_features=384, bias=False)
              (dropout): Dropout(p=0.2, inplace=False)
            )
          )
        )
        (ln_f): LayerNorm()
      )
      (lm_head): Linear(in_features=384, out_features=65, bias=False)
    )
    [2025-07-24 17:54:35,019170][I][wordplay/trainer:348] • self.grad_scaler=None
    [2025-07-24 17:54:35,019844][I][wordplay/trainer:349] • self.model_engine=GPT(
      (transformer): ModuleDict(
        (wte): Embedding(65, 384)
        (wpe): Embedding(128, 384)
        (drop): Dropout(p=0.2, inplace=False)
        (h): ModuleList(
          (0-5): 6 x Block(
            (ln_1): LayerNorm()
            (attn): CausalSelfAttention(
              (c_attn): Linear(in_features=384, out_features=1152, bias=False)
              (c_proj): Linear(in_features=384, out_features=384, bias=False)
              (attn_dropout): Dropout(p=0.2, inplace=False)
              (resid_dropout): Dropout(p=0.2, inplace=False)
            )
            (ln_2): LayerNorm()
            (mlp): MLP(
              (c_fc): Linear(in_features=384, out_features=1536, bias=False)
              (act_fn): GELU(approximate='none')
              (c_proj): Linear(in_features=1536, out_features=384, bias=False)
              (dropout): Dropout(p=0.2, inplace=False)
            )
          )
        )
        (ln_f): LayerNorm()
      )
      (lm_head): Linear(in_features=384, out_features=65, bias=False)
    )
    [2025-07-24 17:54:35,022322][I][wordplay/trainer:350] • self.optimizer=AdamW (
    Parameter Group 0
        amsgrad: False
        betas: (0.9, 0.99)
        capturable: False
        decoupled_weight_decay: True
        differentiable: False
        eps: 1e-08
        foreach: None
        fused: None
        lr: 0.001
        maximize: False
        weight_decay: 0.1

    Parameter Group 1
        amsgrad: False
        betas: (0.9, 0.99)
        capturable: False
        decoupled_weight_decay: True
        differentiable: False
        eps: 1e-08
        foreach: None
        fused: None
        lr: 0.001
        maximize: False
        weight_decay: 0.0
    )

## Prompt (**prior** to training)

``` python
query = "What is an LLM?"
outputs = trainer.evaluate(
    query,
    num_samples=1,
    max_new_tokens=256,
    top_k=16,
    display=False
)
logger.info(f"['prompt']: '{query}'")
logger.info("['response']:\n\n" + fr"{outputs['0']['raw']}")
```

    [2025-07-24 17:54:38,888093][I][ipykernel_80295/3496000222:9:ezpz.log] ['prompt']: 'What is an LLM?'
    [2025-07-24 17:54:38,889260][I][ipykernel_80295/3496000222:10:ezpz.log] ['response']:

    What is an LLM?A,,osy'exx.ff.fpppxv;;'vt3QjYhhvvYAhowQwwQ,eqeqG;X.YqqQSZQWLsyccccj:ZhaooxkkcfkZ
    ffop- f,hqWl
    oocpppUqAQ;cc''bQqcWAttrqerrwyqqsrqttqYeqWQs'tottcqestbqbbrpWbWYAppppBqfhcqqYqqM?qttqQU'gYe?A..'S'rtppW'fJf;??qn.pwrrrqqfA;!!A,,,AtqqqqbW;bSoW;;?;;;qQ;;cIA.'M;''g

## Train Model

|  name  |         description         |
|:------:|:---------------------------:|
| `step` |    Current training step    |
| `loss` |         Loss value          |
|  `dt`  |  Time per step (in **ms**)  |
| `sps`  |     Samples per second      |
| `mtps` |  (million) Tokens per sec   |
| `mfu`  | Model Flops utilization[^1] |

^legend: \#tbl-legend

``` python
trainer.config.device_type
```

    'cpu'

``` python
from rich import print

print(trainer.model)
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">GPT</span><span style="font-weight: bold">(</span>
  <span style="font-weight: bold">(</span>transformer<span style="font-weight: bold">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">ModuleDict</span><span style="font-weight: bold">(</span>
    <span style="font-weight: bold">(</span>wte<span style="font-weight: bold">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Embedding</span><span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">65</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">384</span><span style="font-weight: bold">)</span>
    <span style="font-weight: bold">(</span>wpe<span style="font-weight: bold">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Embedding</span><span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">128</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">384</span><span style="font-weight: bold">)</span>
    <span style="font-weight: bold">(</span>drop<span style="font-weight: bold">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Dropout</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">p</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.2</span>, <span style="color: #808000; text-decoration-color: #808000">inplace</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span><span style="font-weight: bold">)</span>
    <span style="font-weight: bold">(</span>h<span style="font-weight: bold">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">ModuleList</span><span style="font-weight: bold">(</span>
      <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span><span style="font-weight: bold">)</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6</span> x <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Block</span><span style="font-weight: bold">(</span>
        <span style="font-weight: bold">(</span>ln_1<span style="font-weight: bold">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">LayerNorm</span><span style="font-weight: bold">()</span>
        <span style="font-weight: bold">(</span>attn<span style="font-weight: bold">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">CausalSelfAttention</span><span style="font-weight: bold">(</span>
          <span style="font-weight: bold">(</span>c_attn<span style="font-weight: bold">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Linear</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">in_features</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">384</span>, <span style="color: #808000; text-decoration-color: #808000">out_features</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1152</span>, <span style="color: #808000; text-decoration-color: #808000">bias</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span><span style="font-weight: bold">)</span>
          <span style="font-weight: bold">(</span>c_proj<span style="font-weight: bold">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Linear</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">in_features</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">384</span>, <span style="color: #808000; text-decoration-color: #808000">out_features</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">384</span>, <span style="color: #808000; text-decoration-color: #808000">bias</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span><span style="font-weight: bold">)</span>
          <span style="font-weight: bold">(</span>attn_dropout<span style="font-weight: bold">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Dropout</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">p</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.2</span>, <span style="color: #808000; text-decoration-color: #808000">inplace</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span><span style="font-weight: bold">)</span>
          <span style="font-weight: bold">(</span>resid_dropout<span style="font-weight: bold">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Dropout</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">p</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.2</span>, <span style="color: #808000; text-decoration-color: #808000">inplace</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span><span style="font-weight: bold">)</span>
        <span style="font-weight: bold">)</span>
        <span style="font-weight: bold">(</span>ln_2<span style="font-weight: bold">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">LayerNorm</span><span style="font-weight: bold">()</span>
        <span style="font-weight: bold">(</span>mlp<span style="font-weight: bold">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">MLP</span><span style="font-weight: bold">(</span>
          <span style="font-weight: bold">(</span>c_fc<span style="font-weight: bold">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Linear</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">in_features</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">384</span>, <span style="color: #808000; text-decoration-color: #808000">out_features</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1536</span>, <span style="color: #808000; text-decoration-color: #808000">bias</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span><span style="font-weight: bold">)</span>
          <span style="font-weight: bold">(</span>act_fn<span style="font-weight: bold">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">GELU</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">approximate</span>=<span style="color: #008000; text-decoration-color: #008000">'none'</span><span style="font-weight: bold">)</span>
          <span style="font-weight: bold">(</span>c_proj<span style="font-weight: bold">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Linear</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">in_features</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1536</span>, <span style="color: #808000; text-decoration-color: #808000">out_features</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">384</span>, <span style="color: #808000; text-decoration-color: #808000">bias</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span><span style="font-weight: bold">)</span>
          <span style="font-weight: bold">(</span>dropout<span style="font-weight: bold">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Dropout</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">p</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.2</span>, <span style="color: #808000; text-decoration-color: #808000">inplace</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span><span style="font-weight: bold">)</span>
        <span style="font-weight: bold">)</span>
      <span style="font-weight: bold">)</span>
    <span style="font-weight: bold">)</span>
    <span style="font-weight: bold">(</span>ln_f<span style="font-weight: bold">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">LayerNorm</span><span style="font-weight: bold">()</span>
  <span style="font-weight: bold">)</span>
  <span style="font-weight: bold">(</span>lm_head<span style="font-weight: bold">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Linear</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">in_features</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">384</span>, <span style="color: #808000; text-decoration-color: #808000">out_features</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">65</span>, <span style="color: #808000; text-decoration-color: #808000">bias</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span><span style="font-weight: bold">)</span>
<span style="font-weight: bold">)</span>
</pre>

## (partial) Training:

We’ll first train for 500 iterations and then evaluate the models
performance on the same prompt:

> What is an LLM?

``` python
trainer.train(train_iters=500)
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-style: italic">                Training Legend                 </span>
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold">        abbr </span>┃<span style="font-weight: bold"> desc                           </span>┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008000; text-decoration-color: #008000">        step </span>│ Current training iteration     │
│<span style="color: #008000; text-decoration-color: #008000">        loss </span>│ Loss value                     │
│<span style="color: #008000; text-decoration-color: #008000">          dt </span>│ Elapsed time per training step │
│<span style="color: #008000; text-decoration-color: #008000">         dtf </span>│ Elapsed time per forward step  │
│<span style="color: #008000; text-decoration-color: #008000">         dtb </span>│ Elapsed time per backward step │
│<span style="color: #008000; text-decoration-color: #008000">         sps </span>│ Samples per second             │
│<span style="color: #008000; text-decoration-color: #008000"> sps_per_gpu </span>│ Samples per second (per GPU)   │
│<span style="color: #008000; text-decoration-color: #008000">         tps </span>│ Tokens per second              │
│<span style="color: #008000; text-decoration-color: #008000"> tps_per_gpu </span>│ Tokens per second (per GPU)    │
│<span style="color: #008000; text-decoration-color: #008000">         mfu </span>│ Model flops utilization        │
└─────────────┴────────────────────────────────┘
</pre>

    [2025-07-24 17:54:42,654855][I][wordplay/trainer:790] ['prompt']: 'What is an LLM?'
    [2025-07-24 17:54:42,655641][I][wordplay/trainer:794] ['response']:

    What is an LLM?wCw'.AAAfxo..'yfAQfppyybvFYerr.MfYZAcLyQQCkkexx-3lllrpMqxkko-rZx3b'3j-ffSSoqq3hhdf'Q''aq'wqqsoKZb'ec3ZAAA;;o,qff..'fArttgbYtturcbcSYrS-Fff'wwwerwPgJ;.e;yY-SpuyeexqYqgQtpMSYqYgbtQqq''';pfsw,';oA;qqeqcckSAo,,rooMgyQha'''fAA..gg;;'ggtSvrupptkeweqqcqqkk-SvYYIv
    [2025-07-24 17:54:48,709407][I][wordplay/trainer:850] step=10 loss=4.28757 dt=0.0199044 dtf=0.0195728 dtb=0.000124875 sps=50.2401 sps_per_gpu=50.2401 tps=6430.73 tps_per_gpu=6430.73 mfu=0.138961
    [2025-07-24 17:54:48,909469][I][wordplay/trainer:850] step=20 loss=4.28569 dt=0.0207186 dtf=0.0203586 dtb=0.0001395 sps=48.2658 sps_per_gpu=48.2658 tps=6178.02 tps_per_gpu=6178.02 mfu=0.138415
    [2025-07-24 17:54:49,110551][I][wordplay/trainer:850] step=30 loss=4.19012 dt=0.0201965 dtf=0.0198634 dtb=0.000123834 sps=49.5135 sps_per_gpu=49.5135 tps=6337.73 tps_per_gpu=6337.73 mfu=0.138268
    [2025-07-24 17:54:49,326006][I][wordplay/trainer:850] step=40 loss=4.26634 dt=0.0215135 dtf=0.020916 dtb=0.000143541 sps=46.4824 sps_per_gpu=46.4824 tps=5949.75 tps_per_gpu=5949.75 mfu=0.137298
    [2025-07-24 17:54:49,551767][I][wordplay/trainer:850] step=50 loss=4.22804 dt=0.0228783 dtf=0.0225127 dtb=0.000140833 sps=43.7096 sps_per_gpu=43.7096 tps=5594.83 tps_per_gpu=5594.83 mfu=0.135658
    [2025-07-24 17:54:52,291517][I][wordplay/trainer:790] ['prompt']: 'What is an LLM?'
    [2025-07-24 17:54:52,292366][I][wordplay/trainer:794] ['response']:

    What is an LLM?fwxx yY'eyffpCx?ZZZ.eevfeesxqQQYoqapxxxsZ
    vrvb'oZ3qoh33roArW;aafAA''f''QYqAob.aqo.Qyyegg'VcqqYbq3AaFskkcAkfvjb'QQtqQfArWA;Qp'k'goWoq;bbrppfQSYy,,,qqqqMsQuAQ'qgoowqqstSpgli-gggggjGG;cttSAA.pYYIoMSYu;QQSv;?gjJf'eQQQ;yg'Mgo-b';ccIffQSqAA'rqqcII?;;'ecWWllc;'';
    [2025-07-24 17:54:57,267387][I][wordplay/trainer:733] Saving checkpoint to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example
    [2025-07-24 17:54:57,268688][I][wordplay/trainer:734] Saving model to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example/model.pth
    [2025-07-24 17:54:57,444739][I][wordplay/configs:141] Appending /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example to /Users/samforeman/projects/saforem2/wordplay/src/ckpts/checkpoints.log
    [2025-07-24 17:54:57,717693][I][wordplay/trainer:850] step=60 loss=4.20216 dt=0.034885 dtf=0.0344769 dtb=0.000154916 sps=28.6656 sps_per_gpu=28.6656 tps=3669.19 tps_per_gpu=3669.19 mfu=0.130021
    [2025-07-24 17:54:58,008482][I][wordplay/trainer:850] step=70 loss=4.20029 dt=0.019858 dtf=0.019447 dtb=0.000133417 sps=50.3574 sps_per_gpu=50.3574 tps=6445.75 tps_per_gpu=6445.75 mfu=0.130948
    [2025-07-24 17:54:58,211383][I][wordplay/trainer:850] step=80 loss=4.14463 dt=0.0192303 dtf=0.0188333 dtb=0.000180541 sps=52.0013 sps_per_gpu=52.0013 tps=6656.17 tps_per_gpu=6656.17 mfu=0.132236
    [2025-07-24 17:54:58,410849][I][wordplay/trainer:850] step=90 loss=4.14377 dt=0.0203684 dtf=0.0200562 dtb=0.000120875 sps=49.0956 sps_per_gpu=49.0956 tps=6284.24 tps_per_gpu=6284.24 mfu=0.132592
    [2025-07-24 17:54:58,618974][I][wordplay/trainer:850] step=100 loss=4.24105 dt=0.0205756 dtf=0.0201675 dtb=0.000136583 sps=48.6013 sps_per_gpu=48.6013 tps=6220.97 tps_per_gpu=6220.97 mfu=0.132776
    [2025-07-24 17:55:00,895566][I][wordplay/trainer:790] ['prompt']: 'What is an LLM?'
    [2025-07-24 17:55:00,896387][I][wordplay/trainer:794] ['response']:

    What is an LLM?f'xfAhf.qYEZQyyoo--AA,QQAAstpMfYhjc'c..MAj'FF,a33lx.adbssxvVhfsMwyQYosoooc'hzgSSrq.vZZZcq33Sk
    ''vaq.w3AmA'..aYjye'ksr'gbvv,,hqb'eSJJm',rSeqfvrrrW;;bZSS:SqeWtttuYgJvkoBggSA'wst:Sur'txx'rSSqbb;;Qq-;.MsooowbqqqnSpBqSosgggtoo'e;''kG;'g-bWWoqetQ''os'q'tptSSSYe;
    [2025-07-24 17:55:04,426792][I][wordplay/trainer:733] Saving checkpoint to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example
    [2025-07-24 17:55:04,427524][I][wordplay/trainer:734] Saving model to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example/model.pth
    [2025-07-24 17:55:04,630771][I][wordplay/configs:141] Appending /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example to /Users/samforeman/projects/saforem2/wordplay/src/ckpts/checkpoints.log
    [2025-07-24 17:55:04,837583][I][wordplay/trainer:850] step=110 loss=4.30091 dt=0.0224427 dtf=0.0220099 dtb=0.000146209 sps=44.5578 sps_per_gpu=44.5578 tps=5703.4 tps_per_gpu=5703.4 mfu=0.131822
    [2025-07-24 17:55:05,058015][I][wordplay/trainer:850] step=120 loss=4.23854 dt=0.0197457 dtf=0.0194168 dtb=0.000121167 sps=50.644 sps_per_gpu=50.644 tps=6482.43 tps_per_gpu=6482.43 mfu=0.132648
    [2025-07-24 17:55:05,266511][I][wordplay/trainer:850] step=130 loss=4.21194 dt=0.0209936 dtf=0.0205983 dtb=0.000138125 sps=47.6336 sps_per_gpu=47.6336 tps=6097.1 tps_per_gpu=6097.1 mfu=0.132558
    [2025-07-24 17:55:05,473749][I][wordplay/trainer:850] step=140 loss=4.30343 dt=0.0218482 dtf=0.0213865 dtb=0.000131083 sps=45.7704 sps_per_gpu=45.7704 tps=5858.62 tps_per_gpu=5858.62 mfu=0.131962
    [2025-07-24 17:55:05,681204][I][wordplay/trainer:850] step=150 loss=4.25562 dt=0.0199942 dtf=0.0196583 dtb=0.000135959 sps=50.0145 sps_per_gpu=50.0145 tps=6401.85 tps_per_gpu=6401.85 mfu=0.1326
    [2025-07-24 17:55:07,646903][I][wordplay/trainer:790] ['prompt']: 'What is an LLM?'
    [2025-07-24 17:55:07,648095][I][wordplay/trainer:794] ['response']:

    What is an LLM?vXvZoQQoLqQewerA'-''.qqQtXxx'V333jo'gQUoojxttYyfQOCCAASc-sseS

    r.GexS-
    Dv'acQqjpwptxxqqZ!!fqzAAf.v3aag;vYgg'fqY:n;QsrkoBQhbYYQQgoMbZg;;cLf..WSSJhppMSkggkkkkooqWWQ'';xheuAA;pppcSQQqq;??ZppBkqeQsgb'SpWbrr;.gSbbqq;;f.t'gIBq;;WtgbW,rWWYAAqttMA''ggQQQnxrrrrh;;!
    [2025-07-24 17:55:11,179247][I][wordplay/trainer:733] Saving checkpoint to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example
    [2025-07-24 17:55:11,180181][I][wordplay/trainer:734] Saving model to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example/model.pth
    [2025-07-24 17:55:11,301394][I][wordplay/configs:141] Appending /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example to /Users/samforeman/projects/saforem2/wordplay/src/ckpts/checkpoints.log
    [2025-07-24 17:55:11,526240][I][wordplay/trainer:850] step=160 loss=4.22457 dt=0.0246899 dtf=0.0240594 dtb=0.000156458 sps=40.5024 sps_per_gpu=40.5024 tps=5184.31 tps_per_gpu=5184.31 mfu=0.130542
    [2025-07-24 17:55:11,809337][I][wordplay/trainer:850] step=170 loss=4.20268 dt=0.0254512 dtf=0.0250961 dtb=0.00012975 sps=39.2909 sps_per_gpu=39.2909 tps=5029.23 tps_per_gpu=5029.23 mfu=0.128356
    [2025-07-24 17:55:12,170531][I][wordplay/trainer:850] step=180 loss=4.23688 dt=0.0281552 dtf=0.0276542 dtb=0.000187584 sps=35.5174 sps_per_gpu=35.5174 tps=4546.23 tps_per_gpu=4546.23 mfu=0.125344
    [2025-07-24 17:55:12,416401][I][wordplay/trainer:850] step=190 loss=4.28941 dt=0.021717 dtf=0.0213064 dtb=0.000130875 sps=46.0468 sps_per_gpu=46.0468 tps=5893.99 tps_per_gpu=5893.99 mfu=0.125546
    [2025-07-24 17:55:12,635920][I][wordplay/trainer:850] step=200 loss=4.25317 dt=0.0238493 dtf=0.0235083 dtb=0.000133083 sps=41.9299 sps_per_gpu=41.9299 tps=5367.03 tps_per_gpu=5367.03 mfu=0.124589
    [2025-07-24 17:55:14,897637][I][wordplay/trainer:790] ['prompt']: 'What is an LLM?'
    [2025-07-24 17:55:14,898486][I][wordplay/trainer:794] ['response']:

    What is an LLM?qervyyf.af3VAAowAoooooBQecAAqrxxxtXptxGQUVVcNYhhhck;;ooc'DaVqLZZZcP'''GGl..ooosZppV!333QqYYfQSYUUoofkm.tpcq'e''3esseeqqe;;!f'sx'MBfQttopp,qccQn3tgQSk-sffQnpSoo'gYpqqQn';qqecAAS'?AAASYf';pMt??pSSpptSbbYj-tWWYQY?gYIfkqg.nn'gqqc'gtqqtS??A'tu?MBBp???qq;;??A,,,
    [2025-07-24 17:55:18,803834][I][wordplay/trainer:733] Saving checkpoint to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example
    [2025-07-24 17:55:18,804686][I][wordplay/trainer:734] Saving model to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example/model.pth
    [2025-07-24 17:55:18,922346][I][wordplay/configs:141] Appending /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example to /Users/samforeman/projects/saforem2/wordplay/src/ckpts/checkpoints.log
    [2025-07-24 17:55:19,136408][I][wordplay/trainer:850] step=210 loss=4.22371 dt=0.0192137 dtf=0.018839 dtb=0.0001295 sps=52.0463 sps_per_gpu=52.0463 tps=6661.92 tps_per_gpu=6661.92 mfu=0.126526
    [2025-07-24 17:55:19,393682][I][wordplay/trainer:850] step=220 loss=4.23227 dt=0.0235128 dtf=0.0229353 dtb=0.000135625 sps=42.53 sps_per_gpu=42.53 tps=5443.84 tps_per_gpu=5443.84 mfu=0.125637
    [2025-07-24 17:55:19,618525][I][wordplay/trainer:850] step=230 loss=4.22308 dt=0.0222687 dtf=0.0218836 dtb=0.000155541 sps=44.9061 sps_per_gpu=44.9061 tps=5747.98 tps_per_gpu=5747.98 mfu=0.125494
    [2025-07-24 17:55:19,843379][I][wordplay/trainer:850] step=240 loss=4.23777 dt=0.0239087 dtf=0.0234878 dtb=0.000178917 sps=41.8257 sps_per_gpu=41.8257 tps=5353.69 tps_per_gpu=5353.69 mfu=0.124513
    [2025-07-24 17:55:20,093168][I][wordplay/trainer:850] step=250 loss=4.24408 dt=0.0233787 dtf=0.0230403 dtb=0.000131125 sps=42.7739 sps_per_gpu=42.7739 tps=5475.06 tps_per_gpu=5475.06 mfu=0.123893
    [2025-07-24 17:55:22,493900][I][wordplay/trainer:790] ['prompt']: 'What is an LLM?'
    [2025-07-24 17:55:22,497852][I][wordplay/trainer:794] ['response']:

    What is an LLM?.rooffAA rW,,aAA'GoA,aUVVcCoGhvZZcd.QEcNAgxvwYa'haccX.aqo?rrQQ;;QbZ '''fc3FqqWk.'oceQ-h!?Yvs'rw--Qc'333-.hq3AwvvcLq','J-w'''rhqWo--;hSQgSqq;?rqYygAA,asso;q33AA'rbv,J-fof'g'SJJ,;ttcqq;'wgybqppaqttof;;;'''qtqaJpuuYf;paeyfhqg''''qWWbwAA-bbQyg'Sqqos''qYrM;a;??
    [2025-07-24 17:55:26,677140][I][wordplay/trainer:733] Saving checkpoint to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example
    [2025-07-24 17:55:26,678170][I][wordplay/trainer:734] Saving model to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example/model.pth
    [2025-07-24 17:55:26,763343][I][wordplay/configs:141] Appending /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example to /Users/samforeman/projects/saforem2/wordplay/src/ckpts/checkpoints.log
    [2025-07-24 17:55:26,981186][I][wordplay/trainer:850] step=260 loss=4.2759 dt=0.021154 dtf=0.0207075 dtb=0.000181583 sps=47.2725 sps_per_gpu=47.2725 tps=6050.88 tps_per_gpu=6050.88 mfu=0.124579
    [2025-07-24 17:55:27,215262][I][wordplay/trainer:850] step=270 loss=4.31702 dt=0.0234355 dtf=0.0229163 dtb=0.000254167 sps=42.6704 sps_per_gpu=42.6704 tps=5461.81 tps_per_gpu=5461.81 mfu=0.123923
    [2025-07-24 17:55:27,486754][I][wordplay/trainer:850] step=280 loss=4.20612 dt=0.0209037 dtf=0.0205321 dtb=0.000152 sps=47.8383 sps_per_gpu=47.8383 tps=6123.3 tps_per_gpu=6123.3 mfu=0.124763
    [2025-07-24 17:55:27,822689][I][wordplay/trainer:850] step=290 loss=4.22943 dt=0.0456563 dtf=0.0451377 dtb=0.000232833 sps=21.9028 sps_per_gpu=21.9028 tps=2803.56 tps_per_gpu=2803.56 mfu=0.118345
    [2025-07-24 17:55:28,107341][I][wordplay/trainer:850] step=300 loss=4.11928 dt=0.021138 dtf=0.0207116 dtb=0.0001965 sps=47.3081 sps_per_gpu=47.3081 tps=6055.43 tps_per_gpu=6055.43 mfu=0.119595
    [2025-07-24 17:55:30,920105][I][wordplay/trainer:790] ['prompt']: 'What is an LLM?'
    [2025-07-24 17:55:30,921207][I][wordplay/trainer:794] ['response']:

    What is an LLM?L3slghC33vfJQO-eBBBv.Y.Sffs,'gxEUAUCQeswPv,ettLWClrrqeZAtLA.''3NsG..''.sAAmebbqYrv''-
    hTkcxhqqVUvvvfv,lxxlAc..3Zpq''Qsk'st;xlneQssssxS;'tt;cb;??rSQ'k--'t::qqnpYbc;nn;WWqqexSe''ftMqYYttttook;;pgSQQcLgycA;;qqbb''aakqrAAk.h''gYbcLLoopqs:sSSAgZQtiAA.'MMsWllpMt
    [2025-07-24 17:55:34,580689][I][wordplay/trainer:733] Saving checkpoint to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example
    [2025-07-24 17:55:34,581811][I][wordplay/trainer:734] Saving model to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example/model.pth
    [2025-07-24 17:55:34,795545][I][wordplay/configs:141] Appending /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example to /Users/samforeman/projects/saforem2/wordplay/src/ckpts/checkpoints.log
    [2025-07-24 17:55:35,038850][I][wordplay/trainer:850] step=310 loss=4.23252 dt=0.0222985 dtf=0.0218827 dtb=0.000137166 sps=44.846 sps_per_gpu=44.846 tps=5740.29 tps_per_gpu=5740.29 mfu=0.12004
    [2025-07-24 17:55:35,269367][I][wordplay/trainer:850] step=320 loss=4.23608 dt=0.0247837 dtf=0.0235267 dtb=0.00103254 sps=40.3491 sps_per_gpu=40.3491 tps=5164.68 tps_per_gpu=5164.68 mfu=0.119196
    [2025-07-24 17:55:35,556365][I][wordplay/trainer:850] step=330 loss=4.25042 dt=0.0244401 dtf=0.0240442 dtb=0.000156125 sps=40.9164 sps_per_gpu=40.9164 tps=5237.3 tps_per_gpu=5237.3 mfu=0.118594
    [2025-07-24 17:55:35,809616][I][wordplay/trainer:850] step=340 loss=4.19956 dt=0.0220765 dtf=0.0216637 dtb=0.000167625 sps=45.2971 sps_per_gpu=45.2971 tps=5798.03 tps_per_gpu=5798.03 mfu=0.119263
    [2025-07-24 17:55:36,068181][I][wordplay/trainer:850] step=350 loss=4.2746 dt=0.0306815 dtf=0.0302594 dtb=0.000167417 sps=32.593 sps_per_gpu=32.593 tps=4171.9 tps_per_gpu=4171.9 mfu=0.116352
    [2025-07-24 17:55:38,271074][I][wordplay/trainer:790] ['prompt']: 'What is an LLM?'
    [2025-07-24 17:55:38,271910][I][wordplay/trainer:794] ['response']:

    What is an LLM?llBZexQZ wwwZrrxxxcqWa vqqxtqK..aHqQqqqecaask..--'Ve'll3fh3k..ttesscU''aUxhSpepBqqepp
    'QQ-;AqfwetpM vSQwbrrZQqa.CAA,,axqbQu''seyex...'';yyfw'gk:SSWQtrrqW''KKpp?ZQU'''tcb?;;;WufBWbb;f'ggYQttSk;?;;;?fA..Sbt;n''rrWqqMeeq;b'k'eMwQQtpufAAqQYAWASSe'qSpqqtLgWoqSk
    [2025-07-24 17:55:41,990543][I][wordplay/trainer:733] Saving checkpoint to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example
    [2025-07-24 17:55:41,992641][I][wordplay/trainer:734] Saving model to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example/model.pth
    [2025-07-24 17:55:42,116243][I][wordplay/configs:141] Appending /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example to /Users/samforeman/projects/saforem2/wordplay/src/ckpts/checkpoints.log
    [2025-07-24 17:55:42,352292][I][wordplay/trainer:850] step=360 loss=4.3276 dt=0.0251687 dtf=0.024809 dtb=0.000136041 sps=39.7319 sps_per_gpu=39.7319 tps=5085.69 tps_per_gpu=5085.69 mfu=0.115706
    [2025-07-24 17:55:42,597954][I][wordplay/trainer:850] step=370 loss=4.15959 dt=0.023457 dtf=0.022943 dtb=0.000152084 sps=42.6313 sps_per_gpu=42.6313 tps=5456.8 tps_per_gpu=5456.8 mfu=0.115927
    [2025-07-24 17:55:42,835779][I][wordplay/trainer:850] step=380 loss=4.21489 dt=0.0267641 dtf=0.0263147 dtb=0.000207084 sps=37.3635 sps_per_gpu=37.3635 tps=4782.53 tps_per_gpu=4782.53 mfu=0.114669
    [2025-07-24 17:55:43,097242][I][wordplay/trainer:850] step=390 loss=4.18483 dt=0.0285168 dtf=0.0279289 dtb=0.000202083 sps=35.067 sps_per_gpu=35.067 tps=4488.58 tps_per_gpu=4488.58 mfu=0.112901
    [2025-07-24 17:55:43,359070][I][wordplay/trainer:850] step=400 loss=4.2439 dt=0.0248223 dtf=0.0243803 dtb=0.000179958 sps=40.2864 sps_per_gpu=40.2864 tps=5156.66 tps_per_gpu=5156.66 mfu=0.112754
    [2025-07-24 17:55:46,632033][I][wordplay/trainer:790] ['prompt']: 'What is an LLM?'
    [2025-07-24 17:55:46,636503][I][wordplay/trainer:794] ['response']:

    What is an LLM?.3YZfxsaskoRbawwqW3fkYfVUB33emX3cxeQ;XAA,E;hqqqAA,VqYoqep.3-S'eh3cPe''bqqQAh
    fSpppp;!cbWA'fff3feNhaAo,Ax.tqq33-33--fCttppaww-gkttttt,,oWbb'glQWb'WWbZexG?b'sWl'tqt?qqQ'M'rhWlfMMe;tc-eqnnfCqYq;'?;t'Mwhqqq'..oooA,rqqfooWkkjGqqqqqq;fs;QYbWkkf',,.SSSbqqqbqeeqff
    [2025-07-24 17:55:50,448107][I][wordplay/trainer:733] Saving checkpoint to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example
    [2025-07-24 17:55:50,449039][I][wordplay/trainer:734] Saving model to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example/model.pth
    [2025-07-24 17:55:50,619845][I][wordplay/configs:141] Appending /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example to /Users/samforeman/projects/saforem2/wordplay/src/ckpts/checkpoints.log
    [2025-07-24 17:55:50,858980][I][wordplay/trainer:850] step=410 loss=4.23287 dt=0.0211274 dtf=0.020664 dtb=0.000238083 sps=47.3319 sps_per_gpu=47.3319 tps=6058.48 tps_per_gpu=6058.48 mfu=0.11457
    [2025-07-24 17:55:51,075403][I][wordplay/trainer:850] step=420 loss=4.27257 dt=0.0203245 dtf=0.0196317 dtb=0.000168583 sps=49.2017 sps_per_gpu=49.2017 tps=6297.82 tps_per_gpu=6297.82 mfu=0.116722
    [2025-07-24 17:55:51,302496][I][wordplay/trainer:850] step=430 loss=4.18557 dt=0.0243618 dtf=0.0237037 dtb=0.00014 sps=41.048 sps_per_gpu=41.048 tps=5254.14 tps_per_gpu=5254.14 mfu=0.116404
    [2025-07-24 17:55:51,542547][I][wordplay/trainer:850] step=440 loss=4.21616 dt=0.0253792 dtf=0.024935 dtb=0.000150666 sps=39.4024 sps_per_gpu=39.4024 tps=5043.51 tps_per_gpu=5043.51 mfu=0.115662
    [2025-07-24 17:55:51,782138][I][wordplay/trainer:850] step=450 loss=4.23928 dt=0.0270445 dtf=0.026571 dtb=0.00015375 sps=36.976 sps_per_gpu=36.976 tps=4732.93 tps_per_gpu=4732.93 mfu=0.114323
    [2025-07-24 17:55:53,951441][I][wordplay/trainer:790] ['prompt']: 'What is an LLM?'
    [2025-07-24 17:55:53,952318][I][wordplay/trainer:794] ['response']:

    What is an LLM?weeQQQ''QQ'evfhQQ;K.AEsWqb..CfC.h;vvx''bTopBe'gWvXffv3ebssW.;?ptdeeep vrr..CCfkqcptyhpwTssWqsAxrqqqehmuZqZ:qeqGGGGauyfxrrAtgSrqWQ,,t;;ppMMgyeqfvfAAqcWYtqqoopepwySkkqggt3bZMqqq;;yybkSJcSQuuurruqqQtttoo''fAqq;;vSJZZZtM''qqM???gWWAAAt??MYYYe;yglAg;up'exuqqWtu
    [2025-07-24 17:55:57,643874][I][wordplay/trainer:733] Saving checkpoint to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example
    [2025-07-24 17:55:57,645202][I][wordplay/trainer:734] Saving model to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example/model.pth
    [2025-07-24 17:55:57,869430][I][wordplay/configs:141] Appending /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example to /Users/samforeman/projects/saforem2/wordplay/src/ckpts/checkpoints.log
    [2025-07-24 17:55:58,090243][I][wordplay/trainer:850] step=460 loss=4.24269 dt=0.0206855 dtf=0.020356 dtb=0.000130125 sps=48.343 sps_per_gpu=48.343 tps=6187.91 tps_per_gpu=6187.91 mfu=0.116262
    [2025-07-24 17:55:58,322941][I][wordplay/trainer:850] step=470 loss=4.26877 dt=0.0215161 dtf=0.0211165 dtb=0.000138208 sps=46.4768 sps_per_gpu=46.4768 tps=5949.03 tps_per_gpu=5949.03 mfu=0.117491
    [2025-07-24 17:55:58,551019][I][wordplay/trainer:850] step=480 loss=4.19188 dt=0.0218725 dtf=0.0215188 dtb=0.000131834 sps=45.7196 sps_per_gpu=45.7196 tps=5852.11 tps_per_gpu=5852.11 mfu=0.118388
    [2025-07-24 17:55:58,775197][I][wordplay/trainer:850] step=490 loss=4.22611 dt=0.020487 dtf=0.0201027 dtb=0.000145833 sps=48.8114 sps_per_gpu=48.8114 tps=6247.86 tps_per_gpu=6247.86 mfu=0.12005
    [2025-07-24 17:55:59,019562][I][wordplay/trainer:850] step=500 loss=4.21804 dt=0.0218035 dtf=0.0214161 dtb=0.000136 sps=45.8642 sps_per_gpu=45.8642 tps=5870.62 tps_per_gpu=5870.62 mfu=0.120731

``` python
import time

query = "What is an LLM?"
t0 = time.perf_counter()
outputs = trainer.evaluate(
    query,
    num_samples=1,
    max_new_tokens=256,
    top_k=16,
    display=False
)
logger.info(f'took: {time.perf_counter() - t0:.4f}s')
logger.info(f"['prompt']: '{query}'")
logger.info("['response']:\n\n" + fr"{outputs['0']['raw']}")
```

    [autoreload of ezpz.log.config failed: Traceback (most recent call last):
      File "/Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/.venv/lib/python3.13/site-packages/IPython/extensions/autoreload.py", line 325, in check
        superreload(m, reload, self.old_objects)
        ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/.venv/lib/python3.13/site-packages/IPython/extensions/autoreload.py", line 580, in superreload
        module = reload(module)
      File "/Users/samforeman/.local/share/uv/python/cpython-3.13.1-macos-aarch64-none/lib/python3.13/importlib/__init__.py", line 129, in reload
        _bootstrap._exec(spec, module)
        ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^
      File "<frozen importlib._bootstrap>", line 866, in _exec
      File "<frozen importlib._bootstrap_external>", line 1022, in exec_module
      File "<frozen importlib._bootstrap_external>", line 1160, in get_code
      File "<frozen importlib._bootstrap_external>", line 1090, in source_to_code
      File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
      File "/Users/samforeman/projects/saforem2/ezpz/src/ezpz/log/config.py", line 184
        "repr.number": {"color": cyan", "bold": False},
                                             ^
    SyntaxError: unterminated string literal (detected at line 184)
    ]

    [2025-07-24 17:56:01,835029][I][ipykernel_80295/1425179755:12:ezpz.log] took: 2.5511s
    [2025-07-24 17:56:01,837534][I][ipykernel_80295/1425179755:13:ezpz.log] ['prompt']: 'What is an LLM?'
    [2025-07-24 17:56:01,838110][I][ipykernel_80295/1425179755:14:ezpz.log] ['response']:

    What is an LLM?fwll

    b3afqbZZI,r oppq3A33QoUUye-fwC'3b3.',A'.hhPlVXXqeQyCCC;xfssc;wTTTTcdGoeehQOCXXXB'KZ--qehoF3AqfqqW
    cQAcceffGG,'fSJpppww,txMgQs;;;?qf'fSSrpcg?s,A'rr,aso?''o'MtQrrSSgqfttggSc''Wb'qA,.Apcbb???;pYYySQ'agggScWQgbqWfqYroffSYSYhqfk''qfAA,sgWlnZ:pt,JynS'gJZes

## Resume Training…

``` python
trainer.train()
```

    [2025-07-24 17:56:04,111964][I][wordplay/trainer:790] ['prompt']: 'What is an LLM?'
    [2025-07-24 17:56:04,113387][I][wordplay/trainer:794] ['response']:

    What is an LLM?wZbbbT'3weew,'foBB.qWWlpwes.qqQevFAA.bbvFF-AkacWWfYhx3fooB'''';vveesppWW
    eeWA3ZZppPZe;dCCvres ;ecc--Ws'cqor,JZVVVCCeepfqqWxApBBBBhh;;JeQhMMss,,wshrhW?BiMWYqqwwwAASSwrrroo,rqtWseMq.Ak'ofA,,'t,,..hh;xx'?sAq';cqxrqWkeMqt'gzAAxhrpqt'g't;?btoseq-pqq'qAtttt,eqrM
    [2025-07-24 17:56:07,751336][I][wordplay/trainer:733] Saving checkpoint to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example
    [2025-07-24 17:56:07,752428][I][wordplay/trainer:734] Saving model to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example/model.pth
    [2025-07-24 17:56:07,943511][I][wordplay/configs:141] Appending /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example to /Users/samforeman/projects/saforem2/wordplay/src/ckpts/checkpoints.log
    [2025-07-24 17:56:08,158381][I][wordplay/trainer:850] step=510 loss=4.25518 dt=0.0211205 dtf=0.0207702 dtb=0.000131667 sps=47.3473 sps_per_gpu=47.3473 tps=6060.45 tps_per_gpu=6060.45 mfu=0.130959
    [2025-07-24 17:56:08,357593][I][wordplay/trainer:850] step=520 loss=4.20906 dt=0.0191938 dtf=0.0188179 dtb=0.000148208 sps=52.1001 sps_per_gpu=52.1001 tps=6668.81 tps_per_gpu=6668.81 mfu=0.132274
    [2025-07-24 17:56:08,573184][I][wordplay/trainer:850] step=530 loss=4.22394 dt=0.0248612 dtf=0.0244687 dtb=0.000169542 sps=40.2234 sps_per_gpu=40.2234 tps=5148.59 tps_per_gpu=5148.59 mfu=0.130172
    [2025-07-24 17:56:08,810466][I][wordplay/trainer:850] step=540 loss=4.23923 dt=0.0239686 dtf=0.0235865 dtb=0.000143917 sps=41.7212 sps_per_gpu=41.7212 tps=5340.31 tps_per_gpu=5340.31 mfu=0.128695
    [2025-07-24 17:56:09,066218][I][wordplay/trainer:850] step=550 loss=4.24928 dt=0.022075 dtf=0.0217351 dtb=0.000131208 sps=45.3 sps_per_gpu=45.3 tps=5798.4 tps_per_gpu=5798.4 mfu=0.128355
    [2025-07-24 17:56:11,121509][I][wordplay/trainer:790] ['prompt']: 'What is an LLM?'
    [2025-07-24 17:56:11,123283][I][wordplay/trainer:794] ['response']:

    What is an LLM?wboG',ZZswPZZhsf'V.h;QrppwAfAa''qWWYYfOOx33fvkkfQ'elccB3kkkm....swevfsssoAkfQss 'f;ehewqs3--seuCeerqfQA,XXqooU;?';QhdI'M;;astc;W;?A;p;p',,'''gosS;;WW?'errs'fwwr''qqWW,w'l;''www''tppwbQWWseSSqYtLtSbQQQ'q;qqM'tbqW,s'r.AAtcbbq-'ttuuA,;;;Q'S;;;ttMglqYetqeSS;Wq
    [2025-07-24 17:56:14,632179][I][wordplay/trainer:733] Saving checkpoint to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example
    [2025-07-24 17:56:14,633097][I][wordplay/trainer:734] Saving model to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example/model.pth
    [2025-07-24 17:56:14,749580][I][wordplay/configs:141] Appending /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example to /Users/samforeman/projects/saforem2/wordplay/src/ckpts/checkpoints.log
    [2025-07-24 17:56:14,960902][I][wordplay/trainer:850] step=560 loss=4.21979 dt=0.020242 dtf=0.0198084 dtb=0.000190834 sps=49.4021 sps_per_gpu=49.4021 tps=6323.47 tps_per_gpu=6323.47 mfu=0.129184
    [2025-07-24 17:56:15,203853][I][wordplay/trainer:850] step=570 loss=4.27896 dt=0.0267543 dtf=0.0258784 dtb=0.000368584 sps=37.3772 sps_per_gpu=37.3772 tps=4784.28 tps_per_gpu=4784.28 mfu=0.126604
    [2025-07-24 17:56:15,439354][I][wordplay/trainer:850] step=580 loss=4.25036 dt=0.0260735 dtf=0.0256463 dtb=0.000160875 sps=38.3531 sps_per_gpu=38.3531 tps=4909.2 tps_per_gpu=4909.2 mfu=0.124552
    [2025-07-24 17:56:15,665945][I][wordplay/trainer:850] step=590 loss=4.30325 dt=0.0233435 dtf=0.0230295 dtb=0.000132958 sps=42.8384 sps_per_gpu=42.8384 tps=5483.32 tps_per_gpu=5483.32 mfu=0.123945
    [2025-07-24 17:56:15,889883][I][wordplay/trainer:850] step=600 loss=4.24977 dt=0.0241436 dtf=0.0237478 dtb=0.0001265 sps=41.4188 sps_per_gpu=41.4188 tps=5301.61 tps_per_gpu=5301.61 mfu=0.123007
    [2025-07-24 17:56:17,918510][I][wordplay/trainer:790] ['prompt']: 'What is an LLM?'
    [2025-07-24 17:56:17,919631][I][wordplay/trainer:794] ['response']:

    What is an LLM?LQ3vvye! wePZ ewbAII''QYUfY.vTcaQlccCfhsZblYe''vS'xqosfoxCx'q33ckkxpppcecZZ-caqAb''fQ-eqb'.AGGGZZ?--s..h.ttppMq3ZQs,e';pwsf..se;;pqtcenr'.nxnqqgbqQYtttM'fSbttcqqqqgYYjjrqfAkkSSSuQqoh'''S;SYYYAG;SSSo'QQQuu;'QSfqo'.tgSggkqWYYbbvqqtuiqrhS;QC'QSrSbWWSJJeuuiWYu
    [2025-07-24 17:56:21,572136][I][wordplay/trainer:733] Saving checkpoint to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example
    [2025-07-24 17:56:21,573005][I][wordplay/trainer:734] Saving model to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example/model.pth
    [2025-07-24 17:56:21,773879][I][wordplay/configs:141] Appending /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example to /Users/samforeman/projects/saforem2/wordplay/src/ckpts/checkpoints.log
    [2025-07-24 17:56:21,979904][I][wordplay/trainer:850] step=610 loss=4.27699 dt=0.0203758 dtf=0.0199587 dtb=0.00015425 sps=49.0778 sps_per_gpu=49.0778 tps=6281.96 tps_per_gpu=6281.96 mfu=0.124281
    [2025-07-24 17:56:22,194937][I][wordplay/trainer:850] step=620 loss=4.2417 dt=0.0228634 dtf=0.0224084 dtb=0.000139167 sps=43.738 sps_per_gpu=43.738 tps=5598.46 tps_per_gpu=5598.46 mfu=0.12395
    [2025-07-24 17:56:22,406209][I][wordplay/trainer:850] step=630 loss=4.1949 dt=0.0216347 dtf=0.0212935 dtb=0.000126333 sps=46.2221 sps_per_gpu=46.2221 tps=5916.43 tps_per_gpu=5916.43 mfu=0.12434
    [2025-07-24 17:56:22,620616][I][wordplay/trainer:850] step=640 loss=4.21554 dt=0.0225131 dtf=0.0221282 dtb=0.00014875 sps=44.4186 sps_per_gpu=44.4186 tps=5685.58 tps_per_gpu=5685.58 mfu=0.124192
    [2025-07-24 17:56:22,835393][I][wordplay/trainer:850] step=650 loss=4.26643 dt=0.0237441 dtf=0.0233749 dtb=0.000142334 sps=42.1158 sps_per_gpu=42.1158 tps=5390.82 tps_per_gpu=5390.82 mfu=0.123422
    [2025-07-24 17:56:24,844977][I][wordplay/trainer:790] ['prompt']: 'What is an LLM?'
    [2025-07-24 17:56:24,846409][I][wordplay/trainer:794] ['response']:

    What is an LLM?qadZ--e'ovTqro'qE'rpAYvrr;qo3AAwUA-sG..qqbaNNyyep;blgWVe''tkaoo,ebqqUAAAAxttmZS.tGlAxxtccZAk'qffhMM;hqcZ
    'rvsoAAtqWtt,'MqWtt'qqqQ--zpttttuq3brqtrrha;WW'eq;cqqqqrrhh-ppq;'SSJrhSYSJqg'',asqqAhdqbv'?Bqqqb',fqSqt'QqAAWAAqqQQQttttIffvqeWYY--?MfSpppMttttBBM'KK..
    [2025-07-24 17:56:28,345184][I][wordplay/trainer:733] Saving checkpoint to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example
    [2025-07-24 17:56:28,345887][I][wordplay/trainer:734] Saving model to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example/model.pth
    [2025-07-24 17:56:28,569056][I][wordplay/configs:141] Appending /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example to /Users/samforeman/projects/saforem2/wordplay/src/ckpts/checkpoints.log
    [2025-07-24 17:56:28,776036][I][wordplay/trainer:850] step=660 loss=4.17238 dt=0.0213663 dtf=0.0209433 dtb=0.000163625 sps=46.8027 sps_per_gpu=46.8027 tps=5990.74 tps_per_gpu=5990.74 mfu=0.124025
    [2025-07-24 17:56:28,997055][I][wordplay/trainer:850] step=670 loss=4.33205 dt=0.0212203 dtf=0.0196397 dtb=0.000166875 sps=47.1247 sps_per_gpu=47.1247 tps=6031.96 tps_per_gpu=6031.96 mfu=0.124657
    [2025-07-24 17:56:29,214447][I][wordplay/trainer:850] step=680 loss=4.17701 dt=0.0223877 dtf=0.02207 dtb=0.000120125 sps=44.6673 sps_per_gpu=44.6673 tps=5717.41 tps_per_gpu=5717.41 mfu=0.124546
    [2025-07-24 17:56:29,424335][I][wordplay/trainer:850] step=690 loss=4.23023 dt=0.0212292 dtf=0.0208896 dtb=0.000124292 sps=47.1049 sps_per_gpu=47.1049 tps=6029.43 tps_per_gpu=6029.43 mfu=0.12512
    [2025-07-24 17:56:29,645094][I][wordplay/trainer:850] step=700 loss=4.19011 dt=0.0217299 dtf=0.0212904 dtb=0.000185583 sps=46.0195 sps_per_gpu=46.0195 tps=5890.5 tps_per_gpu=5890.5 mfu=0.125337
    [2025-07-24 17:56:31,879979][I][wordplay/trainer:790] ['prompt']: 'What is an LLM?'
    [2025-07-24 17:56:31,880848][I][wordplay/trainer:794] ['response']:

    What is an LLM?lrvqqrafQEsA,hrccZZ;'rrkf'c x'Xxqad.SSxtaV!XQUxv;a.'g
    Zto..herovV-qA'K;aZs3ecAq vqq.!c'fos,ssAAcqfop-;AA.Ag.WYYvvqttxW,,eq;;..Mww';QtMMgqeeqYYppppp;;..MW'tqYf.ff';ccWYrrS'SAsSohegQrr'rhWSASpgj'.A;;.eqqqqqeWWofYQYtcb'Q;;;tttuqcgk;.t3tSbYhhouI;ppp;tSfvgQSuSq
    [2025-07-24 17:56:35,393311][I][wordplay/trainer:733] Saving checkpoint to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example
    [2025-07-24 17:56:35,395178][I][wordplay/trainer:734] Saving model to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example/model.pth
    [2025-07-24 17:56:35,605681][I][wordplay/configs:141] Appending /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example to /Users/samforeman/projects/saforem2/wordplay/src/ckpts/checkpoints.log
    [2025-07-24 17:56:35,826629][I][wordplay/trainer:850] step=710 loss=4.25752 dt=0.023553 dtf=0.0228638 dtb=0.00033675 sps=42.4575 sps_per_gpu=42.4575 tps=5434.56 tps_per_gpu=5434.56 mfu=0.124547
    [2025-07-24 17:56:36,061245][I][wordplay/trainer:850] step=720 loss=4.22592 dt=0.0209667 dtf=0.020581 dtb=0.000175375 sps=47.6947 sps_per_gpu=47.6947 tps=6104.92 tps_per_gpu=6104.92 mfu=0.125284
    [2025-07-24 17:56:36,278396][I][wordplay/trainer:850] step=730 loss=4.18346 dt=0.0203766 dtf=0.0199708 dtb=0.000166375 sps=49.0758 sps_per_gpu=49.0758 tps=6281.71 tps_per_gpu=6281.71 mfu=0.12633
    [2025-07-24 17:56:36,503093][I][wordplay/trainer:850] step=740 loss=4.22937 dt=0.022345 dtf=0.0219809 dtb=0.000145542 sps=44.7527 sps_per_gpu=44.7527 tps=5728.34 tps_per_gpu=5728.34 mfu=0.126075
    [2025-07-24 17:56:36,719860][I][wordplay/trainer:850] step=750 loss=4.22004 dt=0.0215283 dtf=0.0210924 dtb=0.000158334 sps=46.4506 sps_per_gpu=46.4506 tps=5945.68 tps_per_gpu=5945.68 mfu=0.126315
    [2025-07-24 17:56:38,790562][I][wordplay/trainer:790] ['prompt']: 'What is an LLM?'
    [2025-07-24 17:56:38,791531][I][wordplay/trainer:794] ['response']:

    What is an LLM?.AvexhjjsAxx3AAAAffyyY'rr.AxZZpaff.yykfAqYEZ
    'koBf''3YYo.hzA,aaqbbZ ttQhhxkeQU'qhqqoqq!!'ffor'f.aZPeG'qW.ttvafA-b??fffvfvYrcL.bWtSS??qtLtQutohdyyppu''rrSqYqc'KKye''''gjjQq'fgJq;;.'gYqrkssW'tp;bqqf.qowqoMM'qQQSqqWssgyttu?qoo'ff''kkSSffAr.MggesgIIBBYeeWqqqqg
    [2025-07-24 17:56:42,245421][I][wordplay/trainer:733] Saving checkpoint to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example
    [2025-07-24 17:56:42,246480][I][wordplay/trainer:734] Saving model to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example/model.pth
    [2025-07-24 17:56:42,477811][I][wordplay/configs:141] Appending /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example to /Users/samforeman/projects/saforem2/wordplay/src/ckpts/checkpoints.log
    [2025-07-24 17:56:42,688784][I][wordplay/trainer:850] step=760 loss=4.16349 dt=0.019823 dtf=0.0194162 dtb=0.000180875 sps=50.4466 sps_per_gpu=50.4466 tps=6457.16 tps_per_gpu=6457.16 mfu=0.127637
    [2025-07-24 17:56:42,913976][I][wordplay/trainer:850] step=770 loss=4.22062 dt=0.0281325 dtf=0.0277775 dtb=0.000127709 sps=35.5461 sps_per_gpu=35.5461 tps=4549.9 tps_per_gpu=4549.9 mfu=0.124705
    [2025-07-24 17:56:43,129983][I][wordplay/trainer:850] step=780 loss=4.16916 dt=0.0227092 dtf=0.0223645 dtb=0.000129708 sps=44.0351 sps_per_gpu=44.0351 tps=5636.49 tps_per_gpu=5636.49 mfu=0.124414
    [2025-07-24 17:56:43,566960][I][wordplay/trainer:850] step=790 loss=4.21405 dt=0.0229462 dtf=0.0222719 dtb=0.000385583 sps=43.5803 sps_per_gpu=43.5803 tps=5578.27 tps_per_gpu=5578.27 mfu=0.124027
    [2025-07-24 17:56:43,791757][I][wordplay/trainer:850] step=800 loss=4.23569 dt=0.0202938 dtf=0.0199111 dtb=0.000145916 sps=49.2762 sps_per_gpu=49.2762 tps=6307.35 tps_per_gpu=6307.35 mfu=0.125254
    [2025-07-24 17:56:45,957291][I][wordplay/trainer:790] ['prompt']: 'What is an LLM?'
    [2025-07-24 17:56:45,958849][I][wordplay/trainer:794] ['response']:

    What is an LLM??.ahoskZqeofpQe'v;.p..hqYwqaarswbbc.ahwbkkA''KyhvX.yp'Vc3;oseo.xeeeaa'WQqfhKKfYqqqf.x33xx--;;;.egMcc-qaaovvKKOsvSpwesfgI;;wwerpMgtcgQsb;uQtggyyptokyy';QCy;;asoW,,Jr''''',AkkfYoAAAAAS::::;;.bWttqeqcbA::gYJJbqgjoBhopwe;.s''ggkk'qk.qkGWYYyqqe;''Sbs'MM;;.qqqqQ
    [2025-07-24 17:56:49,752195][I][wordplay/trainer:733] Saving checkpoint to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example
    [2025-07-24 17:56:49,753193][I][wordplay/trainer:734] Saving model to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example/model.pth
    [2025-07-24 17:56:49,978804][I][wordplay/configs:141] Appending /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example to /Users/samforeman/projects/saforem2/wordplay/src/ckpts/checkpoints.log
    [2025-07-24 17:56:50,191980][I][wordplay/trainer:850] step=810 loss=4.22317 dt=0.0278166 dtf=0.0274366 dtb=0.000154042 sps=35.9498 sps_per_gpu=35.9498 tps=4601.57 tps_per_gpu=4601.57 mfu=0.122672
    [2025-07-24 17:56:50,446777][I][wordplay/trainer:850] step=820 loss=4.24584 dt=0.0270907 dtf=0.0266571 dtb=0.000134625 sps=36.9131 sps_per_gpu=36.9131 tps=4724.87 tps_per_gpu=4724.87 mfu=0.120615
    [2025-07-24 17:56:50,683011][I][wordplay/trainer:850] step=830 loss=4.1855 dt=0.0213414 dtf=0.0208322 dtb=0.000287459 sps=46.8573 sps_per_gpu=46.8573 tps=5997.74 tps_per_gpu=5997.74 mfu=0.121514
    [2025-07-24 17:56:50,950210][I][wordplay/trainer:850] step=840 loss=4.24083 dt=0.0290908 dtf=0.0282269 dtb=0.00046325 sps=34.3751 sps_per_gpu=34.3751 tps=4400.01 tps_per_gpu=4400.01 mfu=0.11887
    [2025-07-24 17:56:51,224708][I][wordplay/trainer:850] step=850 loss=4.23785 dt=0.0261763 dtf=0.0258001 dtb=0.000143833 sps=38.2024 sps_per_gpu=38.2024 tps=4889.91 tps_per_gpu=4889.91 mfu=0.11755
    [2025-07-24 17:56:53,669992][I][wordplay/trainer:790] ['prompt']: 'What is an LLM?'
    [2025-07-24 17:56:53,671447][I][wordplay/trainer:794] ['response']:

    What is an LLM?A;QfqrqQ'xxx'aa.hh3vv''wwossqZse'rxfQsseh'.evrpMq''.xxTUeQ'''rqqaxfxtcbqcf3qq3jZbvcepwA,,,ff'hpqcpcA-A'rv::errrvbbZ:pc-qycSScWlbQYhhwwAA-SQCgl;bbrpbSrrrrqqqqq''rWqqtcAkYyqgYtxttttbkkqQWWqaqqqkkk,'qqexrrWSSqyyYj'SyyQYQQ,q''p'---p''tcqzhhhpqWfs.p'foBqqQt::eu
    [2025-07-24 17:56:57,438780][I][wordplay/trainer:733] Saving checkpoint to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example
    [2025-07-24 17:56:57,439768][I][wordplay/trainer:734] Saving model to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example/model.pth
    [2025-07-24 17:56:57,625515][I][wordplay/configs:141] Appending /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example to /Users/samforeman/projects/saforem2/wordplay/src/ckpts/checkpoints.log
    [2025-07-24 17:56:57,839383][I][wordplay/trainer:850] step=860 loss=4.20116 dt=0.0196585 dtf=0.0192937 dtb=0.000161375 sps=50.8687 sps_per_gpu=50.8687 tps=6511.19 tps_per_gpu=6511.19 mfu=0.119865
    [2025-07-24 17:56:58,069010][I][wordplay/trainer:850] step=870 loss=4.22428 dt=0.0191218 dtf=0.0187283 dtb=0.000127417 sps=52.2962 sps_per_gpu=52.2962 tps=6693.92 tps_per_gpu=6693.92 mfu=0.122343
    [2025-07-24 17:56:58,312649][I][wordplay/trainer:850] step=880 loss=4.22977 dt=0.0249633 dtf=0.0246209 dtb=0.000129417 sps=40.0589 sps_per_gpu=40.0589 tps=5127.54 tps_per_gpu=5127.54 mfu=0.121189
    [2025-07-24 17:56:58,550794][I][wordplay/trainer:850] step=890 loss=4.22047 dt=0.0209761 dtf=0.0206669 dtb=0.000117875 sps=47.6732 sps_per_gpu=47.6732 tps=6102.18 tps_per_gpu=6102.18 mfu=0.122256
    [2025-07-24 17:56:58,770797][I][wordplay/trainer:850] step=900 loss=4.35563 dt=0.0222012 dtf=0.0218146 dtb=0.000150958 sps=45.0426 sps_per_gpu=45.0426 tps=5765.45 tps_per_gpu=5765.45 mfu=0.122489
    [2025-07-24 17:57:01,310302][I][wordplay/trainer:790] ['prompt']: 'What is an LLM?'
    [2025-07-24 17:57:01,311647][I][wordplay/trainer:794] ['response']:

    What is an LLM?wwPA'eeew-3ZAjRwqs33eafCq'ax..xcxc''awA',bsettcCvCqqq33A-.bsor.awQfJ$  3a-3b U' Zq3gQQf',,AqGZ fhhPwU.vfCC.xpqvr.SkkofxsyQrrs';'kGs,rMse''rppb'qqfoktM'qo,qqSqgW,etM'M??Z;auYfSSo??gg'sSvSQQqfftcb;;;;pWQSffttqgQSSSkllbrqqaw,'SqqYQ;;;pqqtpBheW;;;.hn'qYyMMesgl
    [2025-07-24 17:57:04,896344][I][wordplay/trainer:733] Saving checkpoint to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example
    [2025-07-24 17:57:04,897152][I][wordplay/trainer:734] Saving model to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example/model.pth
    [2025-07-24 17:57:05,074476][I][wordplay/configs:141] Appending /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example to /Users/samforeman/projects/saforem2/wordplay/src/ckpts/checkpoints.log
    [2025-07-24 17:57:05,278672][I][wordplay/trainer:850] step=910 loss=4.19569 dt=0.020175 dtf=0.0197348 dtb=0.00017525 sps=49.5662 sps_per_gpu=49.5662 tps=6344.47 tps_per_gpu=6344.47 mfu=0.12395
    [2025-07-24 17:57:05,480714][I][wordplay/trainer:850] step=920 loss=4.23206 dt=0.0193145 dtf=0.0189184 dtb=0.000174875 sps=51.7746 sps_per_gpu=51.7746 tps=6627.15 tps_per_gpu=6627.15 mfu=0.125875
    [2025-07-24 17:57:05,683914][I][wordplay/trainer:850] step=930 loss=4.29058 dt=0.0206599 dtf=0.0202852 dtb=0.000153959 sps=48.4029 sps_per_gpu=48.4029 tps=6195.57 tps_per_gpu=6195.57 mfu=0.126676
    [2025-07-24 17:57:05,907921][I][wordplay/trainer:850] step=940 loss=4.211 dt=0.0223109 dtf=0.0218805 dtb=0.000188042 sps=44.8212 sps_per_gpu=44.8212 tps=5737.11 tps_per_gpu=5737.11 mfu=0.126405
    [2025-07-24 17:57:06,156551][I][wordplay/trainer:850] step=950 loss=4.18626 dt=0.0282144 dtf=0.0278702 dtb=0.000128333 sps=35.4429 sps_per_gpu=35.4429 tps=4536.69 tps_per_gpu=4536.69 mfu=0.123568
    [2025-07-24 17:57:08,598808][I][wordplay/trainer:790] ['prompt']: 'What is an LLM?'
    [2025-07-24 17:57:08,601493][I][wordplay/trainer:794] ['response']:

    What is an LLM?YfQooooRx3xccaHCvj3gllexpjGG,wUxe'oOf.smxxxrq-jj'kxxrkc3fkkeQZZe''YR'JhrZZAcowccpqA,QUJZpcAkkGGGqp--.v'appbYYbeeqbbZrk'MBfq-srksqYee'QQt'J',qWqt;qkGWbrrtqJ-'pa'ggjJSq--'sf'..;''aqfpfx'Sbbq3tooMbb?',AA-AW'MqAAk;ccAGqQqaA;WQhMSq;cffho,eWohpWott3jj---s;?ggIIS
    [2025-07-24 17:57:12,562033][I][wordplay/trainer:733] Saving checkpoint to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example
    [2025-07-24 17:57:12,563559][I][wordplay/trainer:734] Saving model to: /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example/model.pth
    [2025-07-24 17:57:12,684673][I][wordplay/configs:141] Appending /Users/samforeman/projects/saforem2/intro-hpc-bootcamp-2025/content/02-llms/7-shakespeare-example to /Users/samforeman/projects/saforem2/wordplay/src/ckpts/checkpoints.log
    [2025-07-24 17:57:12,889556][I][wordplay/trainer:850] step=960 loss=4.225 dt=0.0188264 dtf=0.0184483 dtb=0.000152584 sps=53.1169 sps_per_gpu=53.1169 tps=6798.96 tps_per_gpu=6798.96 mfu=0.125903
    [2025-07-24 17:57:13,099316][I][wordplay/trainer:850] step=970 loss=4.17741 dt=0.0200467 dtf=0.0196303 dtb=0.000136625 sps=49.8835 sps_per_gpu=49.8835 tps=6385.09 tps_per_gpu=6385.09 mfu=0.12711
    [2025-07-24 17:57:13,326412][I][wordplay/trainer:850] step=980 loss=4.1707 dt=0.0205021 dtf=0.0200779 dtb=0.000143667 sps=48.7754 sps_per_gpu=48.7754 tps=6243.26 tps_per_gpu=6243.26 mfu=0.12789
    [2025-07-24 17:57:13,590032][I][wordplay/trainer:850] step=990 loss=4.1891 dt=0.027944 dtf=0.0275755 dtb=0.000144 sps=35.7859 sps_per_gpu=35.7859 tps=4580.6 tps_per_gpu=4580.6 mfu=0.124999
    [2025-07-24 17:57:13,884599][I][wordplay/trainer:850] step=1000 loss=4.2423 dt=0.0267193 dtf=0.026294 dtb=0.000146208 sps=37.4261 sps_per_gpu=37.4261 tps=4790.54 tps_per_gpu=4790.54 mfu=0.122851

## Evaluate Model

``` python
import time

query = "What is an LLM?"
t0 = time.perf_counter()
outputs = trainer.evaluate(
    query,
    num_samples=1,
    max_new_tokens=256,
    top_k=2,
    display=False
)
logger.info(f'took: {time.perf_counter() - t0:.4f}s')
logger.info(f"['prompt']: '{query}'")
logger.info("['response']:\n\n" + fr"{outputs['0']['raw']}")
```

    [2025-07-24 17:57:16,962407][I][ipykernel_80295/582817405:12:ezpz.log] took: 2.5928s
    [2025-07-24 17:57:16,965232][I][ipykernel_80295/582817405:13:ezpz.log] ['prompt']: 'What is an LLM?'
    [2025-07-24 17:57:16,966251][I][ipykernel_80295/582817405:14:ezpz.log] ['response']:

    What is an LLM?ZxxA---'aaaaeeewAAAAA'''qqqqqqqqqqqqaeeqqqqqq''333qqAAA33akkk''qqqqqorrrrrrrrrrqqqqqqq.qe333aaaqqqqqf..qqqqqqq3333333-qqqqbbb''ggSSpMMMqqqqMMqqqqqqqqWW;?;?;?;???;;??MMMM;;;;;;??;;;;;;;;''''';??qqqqqqqW;;''''''''''''';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;'tttttMM

[^1]: in units of A100 `bfloat16` peak FLOPS
