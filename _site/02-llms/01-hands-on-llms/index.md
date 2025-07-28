# Hands On: Introduction to Large Language Models (LLMs)
Sam Foreman
2025-07-23

<link rel="preconnect" href="https://fonts.googleapis.com">

- [Outline](#outline)
- [Modeling Sequential Data](#modeling-sequential-data)
- [Scientific sequential data modeling
  examples](#scientific-sequential-data-modeling-examples)
  - [Nucleic acid sequences + genomic
    data](#nucleic-acid-sequences--genomic-data)
  - [Protein sequences](#protein-sequences)
  - [Other applications](#other-applications)
- [Overview of Language models](#overview-of-language-models)
  - [Transformers](#transformers)
- [Coding example of LLMs in action!](#coding-example-of-llms-in-action)
- [🧑‍💻 Hands On](#technologist-hands-on)
- [Logging Tests](#logging-tests)
- [Back to our regularly scheduled
  program…](#back-to-our-regularly-scheduled-program)
- [What’s going on under the hood?](#whats-going-on-under-the-hood)
- [Tokenization and embedding of sequential
  data](#tokenization-and-embedding-of-sequential-data)
  - [Example of tokenization](#example-of-tokenization)
  - [Token embedding:](#token-embedding)
- [Transformer Model Architecture](#transformer-model-architecture)
  - [Attention mechanisms](#attention-mechanisms)
- [Pipeline using HuggingFace](#pipeline-using-huggingface)
  - [1. Setting up a prompt](#1-setting-up-a-prompt)
  - [2. Loading Pretrained Models](#2-loading-pretrained-models)
  - [3. Loading in the tokenizer and tokenizing input
    text](#3-loading-in-the-tokenizer-and-tokenizing-input-text)
  - [4. Performing inference and
    interpreting](#4-performing-inference-and-interpreting)
  - [Saving and loading models](#saving-and-loading-models)
- [Model Hub](#model-hub)
- [Recommended reading](#recommended-reading)
- [Homework](#homework)

<a href="https://colab.research.google.com/github/argonne-lcf/ai-science-training-series/blob/main/04_intro_to_llms/IntroLLMs.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

> [!NOTE]
>
> ### Authors
>
> Content in this notebook is modified from content originally written
> by:
>
> - Archit Vasan, Huihuo Zheng, Marieme Ngom, Bethany Lusch, Taylor
>   Childers, Venkat Vishwanath
>
> Inspiration from the blog posts “The Illustrated Transformer” and “The
> Illustrated GPT2” by Jay Alammar, highly recommended reading.

## Outline

Although the name “language models” is derived from Natural Language
Processing, the models used in these approaches can be applied to
diverse scientific applications as illustrated below.

During this session I will cover:

1.  Scientific applications for language models
2.  General overview of Transformers
3.  Tokenization
4.  Model Architecture
5.  Pipeline using HuggingFace
6.  Model loading

## Modeling Sequential Data

Sequences are variable-length lists with data in subsequent iterations
that depends on previous iterations (or tokens).

Mathematically:

A sequence is a list of tokens:

$$T = [T_1, T_2, T_3,...,T_N]$$

where each token within the list depends on the others with a particular
probability:

$$P(t_N | t_{N-1}, ..., t_3, t_2, t_1)$$

The purpose of sequential modeling is to learn these probabilities for
possible tokens in a distribution to perform various tasks including:

- Sequence generation based on a prompt
- Language translation (e.g. English –\> French)
- Property prediction (predicting a property based on an entire
  sequence)
- Identifying mistakes or missing elements in sequential data

## Scientific sequential data modeling examples

### Nucleic acid sequences + genomic data

<div id="fig-RNA-codons">

![](https://github.com/architvasan/ai_science_local/blob/main/images/RNA-codons.svg.png?raw=1)

Figure 1: RNA Codons

</div>

Nucleic acid sequences can be used to predict translation of proteins,
mutations, and gene expression levels.

Here is an image of GenSLM. This is a language model developed by
Argonne researchers that can model genomic information in a single
model. It was shown to model the evolution of SARS-COV2 without
expensive experiments.

<div id="fig-genslm">

![](https://github.com/architvasan/ai_science_local/blob/main/images/genslm.png?raw=1)

Figure 2: Genomic Scale Language Models (GenSLM) [Zvyagin et. al 2022.
BioRXiv](https://www.biorxiv.org/content/10.1101/2022.10.10.511571v1)

</div>

### Protein sequences

Protein sequences can be used to predict folding structure,
protein-protein interactions, chemical/binding properties, protein
function and many more properties.

<div id="fig-protein-structure">

![](https://github.com/architvasan/ai_science_local/blob/main/images/Protein-Structure-06.png?raw=1)

Figure 3: Protein Structure

</div>

<div id="fig-esmfold">

![](https://github.com/argonne-lcf/ai-science-training-series/blob/main/04_intro_to_llms/images/ESMFold.png?raw=1)

Figure 4: ESMFold [Lin et. al. 2023.
Science](https://www.science.org/doi/10.1126/science.ade2574)

</div>

### Other applications

- Biomedical text
- SMILES strings
- Weather predictions
- Interfacing with simulations such as molecular dynamics simulation

## Overview of Language models

We will now briefly talk about the progression of language models.

### Transformers

The most common LMs base their design on the Transformer architecture
that was introduced in 2017 in the “Attention is all you need” paper.

<div id="fig-attention-is-all-you-need">

![](https://github.com/architvasan/ai_science_local/blob/main/images/attention_is_all_you_need.png?raw=1)

Figure 5: Attention is all you need [Vaswani 2017. Advances in Neural
Information Processing Systems](https://arxiv.org/pdf/1706.03762)

</div>

Since then a multitude of LLM architectures have been designed.

<div id="fig-ch1-transformers">

![](https://github.com/architvasan/ai_science_local/blob/main/images/en_chapter1_transformers_chrono.svg?raw=1)

Figure 6: Transformers, chronologically

</div>

[HuggingFace NLP
Course](https://huggingface.co/learn/nlp-course/chapter1/4)

## Coding example of LLMs in action!

Let’s look at an example of running inference with a LLM as a block box
to generate text given a prompt and we will also initiate a training
loop for an LLM

Here, we will use the `transformers` library which is as part of
HuggingFace, a repository of different models, tokenizers and
information on how to apply these models

> [!WARNING]
>
> ### 🦜 Stochastic Parrots
>
> **Warning**: *Large Language Models are only as good as their training
> data*.
>
> They have no ethics, judgement, or editing ability.
>
> We will be using some pretrained models from Hugging Face which used
> wide samples of internet hosted text.
>
> The datasets have not been strictly filtered to restrict all malign
> content so the generated text may be surprisingly dark or
> questionable.
>
> They do not reflect our core values and are only used for
> demonstration purposes.

<details closed>

<summary>

<h2>

🏃‍♂️ Running @ ALCF
</h2>

</summary>

- If running this notebook on any of the ALCF machines, be sure to:

  ``` python
  import os
  os.environ["HTTP_PROXY"]="proxy.alcf.anl.gov:3128"
  os.environ["HTTPS_PROXY"]="proxy.alcf.anl.gov:3128"
  os.environ["http_proxy"]="proxy.alcf.anl.gov:3128"
  os.environ["https_proxy"]="proxy.alcf.anl.gov:3128"
  os.environ["ftp_proxy"]="proxy.alcf.anl.gov:3128"
  ```

</details>

## 🧑‍💻 Hands On

``` python
#!pip install transformers
#!pip install pandas
#!pip install torch
```

``` python
%load_ext autoreload
%autoreload 2
```

## Logging Tests

``` python
import logging
import sys

#logging.basicConfig(
#    level=logging.INFO
#    # format='[{%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
#    # handlers=[
#    #     logging.FileHandler(filename='tmp5a.log'),
#    #     logging.StreamHandler(sys.stdout)
#    # ]
#)

# Test
#logger = logging.getLogger('NORMAL')
#logger.debug('This message should go to the log file and to the console')
#logger.info('So should this')
#logger.warning('And this, too')
basic_logger = logging.getLogger('basic')
basic_logger.propagate = False
basic_logger.addHandler(logging.StreamHandler(sys.stdout))
basic_logger.setLevel("INFO")
basic_logger.info("Basic logger")
```

    Basic logger

``` python
import logging
# import ezpz
from rich.console import Console
from rich.logging import RichHandler
```

``` python
# logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
rich_logger = logging.getLogger("rich")
rich_logger.propagate = False
console = Console(color_system="truecolor", soft_wrap=True, log_time=True, log_path=False)  # , theme=ezpz.log.get_theme())
rich_handler = RichHandler(level="INFO", console=console, show_path=False, )
rich_logger.handlers.clear()
rich_logger.addHandler(rich_handler)
rich_logger.setLevel("INFO")
rich_logger.info("Rich logger")
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[07/26/25 17:19:08] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Rich logger                                                                           
</pre>

``` python
from rich.text import Text

t = Text("abc")
```

``` python
t
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">abc
</pre>

``` python
from ezpz.log.handler import EzpzHandler
ezpz_logger = logging.getLogger("ezpz")
ezpz_logger.propagate = False
ezpz_logger.handlers.clear()
#ezpz_console = ezpz.log.get_console()
ezpz_handler = EzpzHandler()
ezpz_logger.setLevel("INFO")
ezpz_logger.addHandler(ezpz_handler)
ezpz_logger.info("ezpz_logger")
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[07/26/25 17:19:18] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Setting logging level to <span style="color: #008000; text-decoration-color: #008000">'INFO'</span> on <span style="color: #008000; text-decoration-color: #008000">'RANK == 0'</span>                         <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">__init__.py:265</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Setting logging level to <span style="color: #008000; text-decoration-color: #008000">'CRITICAL'</span> on all others <span style="color: #008000; text-decoration-color: #008000">'RANK != 0'</span>          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">__init__.py:266</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-07-26 </span><span style="color: #bfbfbf; text-decoration-color: #bfbfbf">17:19:18,068891</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_92725</span>/<span style="color: #000080; text-decoration-color: #000080">3256268672</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">9</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]ezpz_logger                                      
</pre>

``` python
def test_logger(logger):
    for i in range(20):
        logger.info(f"i: {i}")
```

``` python
for l in [basic_logger, rich_logger, ezpz_logger]:
    print('\n\n' + 80 * '-' +'\n\n')
    print(f"Using logger={l}")
    test_logger(l)
```



    --------------------------------------------------------------------------------


    Using logger=<Logger basic (INFO)>
    i: 0g logger=<Logger basic (INFO)>

    1g logger=<Logger basic (INFO)>


    i: 2ogger=<Logger basic (INFO)>



    3ogger=<Logger basic (INFO)>




    i: 4er=<Logger basic (INFO)>





    5er=<Logger basic (INFO)>






    i: 6<Logger basic (INFO)>







    7<Logger basic (INFO)>








    i: 8gger basic (INFO)>









    9gger basic (INFO)>










    i: 10 basic (INFO)>











    1 basic (INFO)>












    i: 12ic (INFO)>













    3ic (INFO)>














    i: 14INFO)>















    5INFO)>
















    i: 16)>

















    7)>


















    i: 18

















    9




















    --------------------------------------------------------------------------------


    Using logger=<Logger rich (INFO)>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[07/26/25 16:27:41] </span><span style="color: #008000; text-decoration-color: #008000">INFO    </span> i: <span style="color: #008080; text-decoration-color: #008080">0</span>                                                                                  
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                    </span><span style="color: #008000; text-decoration-color: #008000">INFO    </span> i: <span style="color: #008080; text-decoration-color: #008080">1</span>                                                                                  
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                    </span><span style="color: #008000; text-decoration-color: #008000">INFO    </span> i: <span style="color: #008080; text-decoration-color: #008080">2</span>                                                                                  
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                    </span><span style="color: #008000; text-decoration-color: #008000">INFO    </span> i: <span style="color: #008080; text-decoration-color: #008080">3</span>                                                                                  
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                    </span><span style="color: #008000; text-decoration-color: #008000">INFO    </span> i: <span style="color: #008080; text-decoration-color: #008080">4</span>                                                                                  
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                    </span><span style="color: #008000; text-decoration-color: #008000">INFO    </span> i: <span style="color: #008080; text-decoration-color: #008080">5</span>                                                                                  
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                    </span><span style="color: #008000; text-decoration-color: #008000">INFO    </span> i: <span style="color: #008080; text-decoration-color: #008080">6</span>                                                                                  
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                    </span><span style="color: #008000; text-decoration-color: #008000">INFO    </span> i: <span style="color: #008080; text-decoration-color: #008080">7</span>                                                                                  
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                    </span><span style="color: #008000; text-decoration-color: #008000">INFO    </span> i: <span style="color: #008080; text-decoration-color: #008080">8</span>                                                                                  
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                    </span><span style="color: #008000; text-decoration-color: #008000">INFO    </span> i: <span style="color: #008080; text-decoration-color: #008080">9</span>                                                                                  
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                    </span><span style="color: #008000; text-decoration-color: #008000">INFO    </span> i: <span style="color: #008080; text-decoration-color: #008080">10</span>                                                                                 
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                    </span><span style="color: #008000; text-decoration-color: #008000">INFO    </span> i: <span style="color: #008080; text-decoration-color: #008080">11</span>                                                                                 
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                    </span><span style="color: #008000; text-decoration-color: #008000">INFO    </span> i: <span style="color: #008080; text-decoration-color: #008080">12</span>                                                                                 
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                    </span><span style="color: #008000; text-decoration-color: #008000">INFO    </span> i: <span style="color: #008080; text-decoration-color: #008080">13</span>                                                                                 
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                    </span><span style="color: #008000; text-decoration-color: #008000">INFO    </span> i: <span style="color: #008080; text-decoration-color: #008080">14</span>                                                                                 
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                    </span><span style="color: #008000; text-decoration-color: #008000">INFO    </span> i: <span style="color: #008080; text-decoration-color: #008080">15</span>                                                                                 
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                    </span><span style="color: #008000; text-decoration-color: #008000">INFO    </span> i: <span style="color: #008080; text-decoration-color: #008080">16</span>                                                                                 
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                    </span><span style="color: #008000; text-decoration-color: #008000">INFO    </span> i: <span style="color: #008080; text-decoration-color: #008080">17</span>                                                                                 
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                    </span><span style="color: #008000; text-decoration-color: #008000">INFO    </span> i: <span style="color: #008080; text-decoration-color: #008080">18</span>                                                                                 
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                    </span><span style="color: #008000; text-decoration-color: #008000">INFO    </span> i: <span style="color: #008080; text-decoration-color: #008080">19</span>                                                                                 
</pre>



    --------------------------------------------------------------------------------


    Using logger=<Logger ezpz (INFO)>
    [2025-07-26 16:27:41,548374][I][ipykernel_70809/243331313:3:ezpz] i: 0
    819][I][ipykernel_70809/243331313:3:ezpz] i: 1

    [2025-07-26 16:27:41,549277][I][ipykernel_70809/243331313:3:ezpz] i: 2
    818][I][ipykernel_70809/243331313:3:ezpz] i: 3

    [2025-07-26 16:27:41,550301][I][ipykernel_70809/243331313:3:ezpz] i: 4
    781][I][ipykernel_70809/243331313:3:ezpz] i: 5

    [2025-07-26 16:27:41,551264][I][ipykernel_70809/243331313:3:ezpz] i: 6
    717][I][ipykernel_70809/243331313:3:ezpz] i: 7

    [2025-07-26 16:27:41,552144][I][ipykernel_70809/243331313:3:ezpz] i: 8
    617][I][ipykernel_70809/243331313:3:ezpz] i: 9

    [2025-07-26 16:27:41,553104][I][ipykernel_70809/243331313:3:ezpz] i: 10
    608][I][ipykernel_70809/243331313:3:ezpz] i: 11

    [2025-07-26 16:27:41,554204][I][ipykernel_70809/243331313:3:ezpz] i: 12
    660][I][ipykernel_70809/243331313:3:ezpz] i: 13

    [2025-07-26 16:27:41,555125][I][ipykernel_70809/243331313:3:ezpz] i: 14
    591][I][ipykernel_70809/243331313:3:ezpz] i: 15

    [2025-07-26 16:27:41,555991][I][ipykernel_70809/243331313:3:ezpz] i: 16
    6384][I][ipykernel_70809/243331313:3:ezpz] i: 17

    [2025-07-26 16:27:41,556719][I][ipykernel_70809/243331313:3:ezpz] i: 18
    7197][I][ipykernel_70809/243331313:3:ezpz] i: 19

## Back to our regularly scheduled program…

``` python
import logging
from ezpz.log.handler import EzpzHandler
logger = logging.getLogger("ezpz")
logger.propagate = False
logger.handlers.clear()
logger.setLevel("INFO")
logger.addHandler(EzpzHandler())
logger.info("logger")
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[07/26/25 19:12:43] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Setting logging level to <span style="color: #008000; text-decoration-color: #008000">'INFO'</span> on <span style="color: #008000; text-decoration-color: #008000">'RANK == 0'</span>                         <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">__init__.py:265</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Setting logging level to <span style="color: #008000; text-decoration-color: #008000">'CRITICAL'</span> on all others <span style="color: #008000; text-decoration-color: #008000">'RANK != 0'</span>          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">__init__.py:266</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-07-26 </span><span style="color: #bfbfbf; text-decoration-color: #bfbfbf">19:12:43,356366</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_18750</span>/<span style="color: #000080; text-decoration-color: #000080">1863777421</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">8</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]logger                                           
</pre>

``` python
import ambivalent

import matplotlib.pyplot as plt
import seaborn as sns

import ezpz
#console = ezpz.log.get_console()
# ezpz_logger = ezpz.get_logger("root")
# ezpz_logger = logging.getLogger("ezpz")
# ezpz_logger.setLevel("INFO")
# ezpz_logger.info("INFO")
#logger = ezpz.get_logger("root")

plt.style.use(ambivalent.STYLES['ambivalent'])
sns.set_context("notebook")
plt.rcParams["figure.figsize"] = [6.4, 4.8]
```

``` python
from transformers import AutoTokenizer,AutoModelForCausalLM, AutoConfig
from transformers import pipeline

input_text = "What is the ALCF?"
generator = pipeline("text-generation", model="gpt2")

pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-3B")
output = generator(input_text, max_length=20, num_return_sequences=2)
```

    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]

``` python
import rich
for i, response in enumerate([i["generated_text"] for i in output]):
    # logger.info(f"Response {i}:\n\t{' '.join(response.split('\n'))}")
    #rich.print(f"Response {i}:\n\t{' '.join(response.split('\n'))}")
    logger.info(f"Response {i}:\n\t{' '.join(response.split('\n'))}")
#logger.info("\n".join([f"Response {i}: {c}"] for i, c in enumerate(outputs)))
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-07-26 </span><span style="color: #bfbfbf; text-decoration-color: #bfbfbf">19:13:16,383194</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_18750</span>/<span style="color: #000080; text-decoration-color: #000080">348035257</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">5</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]Response <span style="color: #008080; text-decoration-color: #008080">0</span>:                                       
        What is the ALCF?  The ALCF is a device that allows you to control a laptop's internal power. That power   
can then be transferred from your computer to the laptop <span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span>depending on the device that you use<span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>. This means that   
you can power up your laptop while it is in use, and it can be controlled by a variety of devices or on your mobile
phone. In order to use the device, you will need to connect an external power source <span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span>such as a wired or wireless  
charger<span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>, which is installed on the laptop.  How can I turn on and off ALCF?  The ALCF is a great way to turn on or
off ALCF. The ALCF will begin a clock cycle that will start when you turn on the computer, and will continue until 
you stop. The ALCF will stop while the computer is on.  Can I turn on and off the computer's ALCF?  The ALCF can   
only be turned on and off when you are on the computer. However, the ALCF does have a few features on its side. The
ALCF can be turned on and off by just holding the ALCF key on the keyboard.  Can I use the ALCF in                 
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-07-26 </span><span style="color: #bfbfbf; text-decoration-color: #bfbfbf">19:13:16,386237</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_18750</span>/<span style="color: #000080; text-decoration-color: #000080">348035257</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">5</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]Response <span style="color: #008080; text-decoration-color: #008080">1</span>:                                       
        What is the ALCF?  ALCF is a class that assists individuals with the ALCF problem. It is a tool for the    
development of skills and understanding of ALCF. The purpose of ALCF is to help individuals to develop a specific  
ALCF problem.  What are the main objectives of ALCF?  ALCF aims to bring together different groups of people. To do
this, it tries to educate and strengthen individuals with the problems of ALCF.  The major goal of ALCF is to      
enable them to achieve their own goals.  What are the main methods for developing a problem?  It requires that the 
organization and the individual use the appropriate skills and tools.  What do the main tasks of ALCF look like?   
At first, ALCF has a few objectives. The first is to provide a means for developing the problem problem.  The      
second is to develop new solutions to the problem problem.  The third is to achieve the goal of developing the ALCF
problem at a higher level. The ALCF problem can be solved in a more efficient manner.  ALCF has several other aims 
such as:  Developing skills/tools for the                                                                          
</pre>

## What’s going on under the hood?

There are two components that are “black-boxes” here:

1.  The method for tokenization
2.  The model that generates novel text.

## Tokenization and embedding of sequential data

Humans can inherently understand language data because they previously
learned phonetic sounds.

Machines don’t have phonetic knowledge so they need to be told how to
break text into standard units to process it.

They use a system called “tokenization”, where sequences of text are
broken into smaller parts, or “tokens”, and then fed as input.

<div id="fig-ai-science-local">

![](https://github.com/architvasan/ai_science_local/blob/main/images/text-processing---machines-vs-humans.png?raw=1)

Figure 7

</div>

Tokenization is a data preprocessing step which transforms the raw text
data into a format suitable for machine learning models. Tokenizers
break down raw text into smaller units called tokens. These tokens are
what is fed into the language models. Based on the type and
configuration of the tokenizer, these tokens can be words, subwords, or
characters.

Types of tokenizers:

1.  Character Tokenizers: Split text into individual characters.
2.  Word Tokenizers: Split text into words based on whitespace or
    punctuation.
3.  Subword Tokenizers: Split text into subword units, such as morphemes
    or character n-grams. Common subword tokenization algorithms
    include:
    1.  Byte-Pair Encoding (BPE),
    2.  SentencePiece,
    3.  WordPiece.

<div id="fig-tokenization-image">

![](https://github.com/architvasan/ai_science_local/blob/main/images/tokenization_image.webp?raw=1)

Figure 8

</div>

[nlpiation](https://nlpiation.medium.com/how-to-use-huggingfaces-transformers-pre-trained-tokenizers-e029e8d6d1fa)

### Example of tokenization

Let’s look at an example of tokenization using byte-pair encoding.

``` python
from transformers import AutoTokenizer

def tokenization_summary(tokenizer, sequence):
    # get the vocabulary
    vocab = tokenizer.vocab
    # Number of entries to print
    n = 10
    # print subset of the vocabulary
    logger.info("Subset of tokenizer.vocab:")
    for i, (token, index) in enumerate(tokenizer.vocab.items()):
        logger.info(f"{token}: {index}")
        if i >= n - 1:
            break
    logger.info(f"Vocab size of the tokenizer = {len(vocab)}")
    logger.info("------------------------------------------")
    # .tokenize chunks the existing sequence into different tokens based on the rules and vocab of the tokenizer.
    tokens = tokenizer.tokenize(sequence)
    logger.info(f"Tokens : {tokens}")
    logger.info("------------------------------------------")

    # .convert_tokens_to_ids or .encode or .tokenize converts the tokens to their corresponding numerical representation.
    #  .convert_tokens_to_ids has a 1-1 mapping between tokens and numerical representation
    # ids = tokenizer.convert_tokens_to_ids(tokens)
    # logger.info("encoded Ids: ", ids)

    # .encode also adds additional information like Start of sequence tokens and End of sequene
    logger.info(f"tokenized sequence : {tokenizer.encode(sequence)}")

    # .tokenizer has additional information about attention_mask.
    # encode = tokenizer(sequence)
    # logger.info("Encode sequence : ", encode)
    # logger.info("------------------------------------------")

    # .decode decodes the ids to raw text
    ids = tokenizer.convert_tokens_to_ids(tokens)
    decode = tokenizer.decode(ids)
    logger.info(f"Decode sequence {decode}")


tokenizer_1  =  AutoTokenizer.from_pretrained("gpt2") # GPT-2 uses "Byte-Pair Encoding (BPE)"

sequence = "Counselor, please adjust your Zoom filter to appear as a human, rather than as a cat"

tokenization_summary(tokenizer_1, sequence)
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-07-26 </span><span style="color: #bfbfbf; text-decoration-color: #bfbfbf">19:14:50,874178</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_18750</span>/<span style="color: #000080; text-decoration-color: #000080">3808680097</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">9</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]Subset of tokenizer.vocab:                       
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-07-26 </span><span style="color: #bfbfbf; text-decoration-color: #bfbfbf">19:14:50,887844</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_18750</span>/<span style="color: #000080; text-decoration-color: #000080">3808680097</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">11</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]iates: <span style="color: #008080; text-decoration-color: #008080">32820</span>                                    
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-07-26 </span><span style="color: #bfbfbf; text-decoration-color: #bfbfbf">19:14:50,889017</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_18750</span>/<span style="color: #000080; text-decoration-color: #000080">3808680097</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">11</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]Ġcalcium: <span style="color: #008080; text-decoration-color: #008080">19700</span>                                 
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-07-26 </span><span style="color: #bfbfbf; text-decoration-color: #bfbfbf">19:14:50,889795</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_18750</span>/<span style="color: #000080; text-decoration-color: #000080">3808680097</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">11</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]Ġbuyers: <span style="color: #008080; text-decoration-color: #008080">14456</span>                                  
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-07-26 </span><span style="color: #bfbfbf; text-decoration-color: #bfbfbf">19:14:50,890517</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_18750</span>/<span style="color: #000080; text-decoration-color: #000080">3808680097</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">11</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]ĠRipple: <span style="color: #008080; text-decoration-color: #008080">40303</span>                                  
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-07-26 </span><span style="color: #bfbfbf; text-decoration-color: #bfbfbf">19:14:50,891199</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_18750</span>/<span style="color: #000080; text-decoration-color: #000080">3808680097</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">11</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]Ġmurdering: <span style="color: #008080; text-decoration-color: #008080">33217</span>                               
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-07-26 </span><span style="color: #bfbfbf; text-decoration-color: #bfbfbf">19:14:50,892065</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_18750</span>/<span style="color: #000080; text-decoration-color: #000080">3808680097</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">11</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]ĠInventory: <span style="color: #008080; text-decoration-color: #008080">35772</span>                               
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-07-26 </span><span style="color: #bfbfbf; text-decoration-color: #bfbfbf">19:14:50,892898</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_18750</span>/<span style="color: #000080; text-decoration-color: #000080">3808680097</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">11</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]Pi: <span style="color: #008080; text-decoration-color: #008080">38729</span>                                       
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-07-26 </span><span style="color: #bfbfbf; text-decoration-color: #bfbfbf">19:14:50,893530</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_18750</span>/<span style="color: #000080; text-decoration-color: #000080">3808680097</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">11</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]ĠHonolulu: <span style="color: #008080; text-decoration-color: #008080">43296</span>                                
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-07-26 </span><span style="color: #bfbfbf; text-decoration-color: #bfbfbf">19:14:50,894128</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_18750</span>/<span style="color: #000080; text-decoration-color: #000080">3808680097</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">11</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]Ġindic: <span style="color: #008080; text-decoration-color: #008080">2699</span>                                    
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-07-26 </span><span style="color: #bfbfbf; text-decoration-color: #bfbfbf">19:14:50,894867</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_18750</span>/<span style="color: #000080; text-decoration-color: #000080">3808680097</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">11</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]ĠCerberus: <span style="color: #008080; text-decoration-color: #008080">42593</span>                                
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-07-26 </span><span style="color: #bfbfbf; text-decoration-color: #bfbfbf">19:14:50,896272</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_18750</span>/<span style="color: #000080; text-decoration-color: #000080">3808680097</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">14</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]Vocab size of the tokenizer = <span style="color: #008080; text-decoration-color: #008080">50257</span>             
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-07-26 </span><span style="color: #bfbfbf; text-decoration-color: #bfbfbf">19:14:50,896922</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_18750</span>/<span style="color: #000080; text-decoration-color: #000080">3808680097</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">15</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]------------------------------------------      
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-07-26 </span><span style="color: #bfbfbf; text-decoration-color: #bfbfbf">19:14:50,898056</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_18750</span>/<span style="color: #000080; text-decoration-color: #000080">3808680097</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">18</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]Tokens : <span style="color: #ff00ff; text-decoration-color: #ff00ff">[</span><span style="color: #008000; text-decoration-color: #008000">'Coun'</span>, <span style="color: #008000; text-decoration-color: #008000">'sel'</span>, <span style="color: #008000; text-decoration-color: #008000">'or'</span>, <span style="color: #008000; text-decoration-color: #008000">','</span>, <span style="color: #008000; text-decoration-color: #008000">'Ġplease'</span>,  
<span style="color: #008000; text-decoration-color: #008000">'Ġadjust'</span>, <span style="color: #008000; text-decoration-color: #008000">'Ġyour'</span>, <span style="color: #008000; text-decoration-color: #008000">'ĠZoom'</span>, <span style="color: #008000; text-decoration-color: #008000">'Ġfilter'</span>, <span style="color: #008000; text-decoration-color: #008000">'Ġto'</span>, <span style="color: #008000; text-decoration-color: #008000">'Ġappear'</span>, <span style="color: #008000; text-decoration-color: #008000">'Ġas'</span>, <span style="color: #008000; text-decoration-color: #008000">'Ġa'</span>, <span style="color: #008000; text-decoration-color: #008000">'Ġhuman'</span>, <span style="color: #008000; text-decoration-color: #008000">','</span>, <span style="color: #008000; text-decoration-color: #008000">'Ġrather'</span>, <span style="color: #008000; text-decoration-color: #008000">'Ġthan'</span>, <span style="color: #008000; text-decoration-color: #008000">'Ġas'</span>,   
<span style="color: #008000; text-decoration-color: #008000">'Ġa'</span>, <span style="color: #008000; text-decoration-color: #008000">'Ġcat'</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">]</span>                                                                                                      
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-07-26 </span><span style="color: #bfbfbf; text-decoration-color: #bfbfbf">19:14:50,899146</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_18750</span>/<span style="color: #000080; text-decoration-color: #000080">3808680097</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">19</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]------------------------------------------      
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-07-26 </span><span style="color: #bfbfbf; text-decoration-color: #bfbfbf">19:14:50,899998</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_18750</span>/<span style="color: #000080; text-decoration-color: #000080">3808680097</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">27</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]tokenized sequence : <span style="color: #ff00ff; text-decoration-color: #ff00ff">[</span><span style="color: #008080; text-decoration-color: #008080">31053</span>, <span style="color: #008080; text-decoration-color: #008080">741</span>, <span style="color: #008080; text-decoration-color: #008080">273</span>, <span style="color: #008080; text-decoration-color: #008080">11</span>, <span style="color: #008080; text-decoration-color: #008080">3387</span>,
<span style="color: #008080; text-decoration-color: #008080">4532</span>, <span style="color: #008080; text-decoration-color: #008080">534</span>, <span style="color: #008080; text-decoration-color: #008080">40305</span>, <span style="color: #008080; text-decoration-color: #008080">8106</span>, <span style="color: #008080; text-decoration-color: #008080">284</span>, <span style="color: #008080; text-decoration-color: #008080">1656</span>, <span style="color: #008080; text-decoration-color: #008080">355</span>, <span style="color: #008080; text-decoration-color: #008080">257</span>, <span style="color: #008080; text-decoration-color: #008080">1692</span>, <span style="color: #008080; text-decoration-color: #008080">11</span>, <span style="color: #008080; text-decoration-color: #008080">2138</span>, <span style="color: #008080; text-decoration-color: #008080">621</span>, <span style="color: #008080; text-decoration-color: #008080">355</span>, <span style="color: #008080; text-decoration-color: #008080">257</span>, <span style="color: #008080; text-decoration-color: #008080">3797</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">]</span>                                  
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-07-26 </span><span style="color: #bfbfbf; text-decoration-color: #bfbfbf">19:14:50,901216</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_18750</span>/<span style="color: #000080; text-decoration-color: #000080">3808680097</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">37</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]Decode sequence Counselor, please adjust your   
Zoom filter to appear as a human, rather than as a cat                                                             
</pre>

### Token embedding:

Words are turned into vectors based on their location within a
vocabulary.

The strategy of choice for learning language structure from tokenized
text is to find a clever way to map each token into a moderate-dimension
vector space, adjusting the mapping so that

Similar, or associated tokens take up residence nearby each other, and
different regions of the space correspond to different position in the
sequence. Such a mapping from token ID to a point in a vector space is
called a token embedding. The dimension of the vector space is often
high (e.g. 1024-dimensional), but much smaller than the vocabulary size
(30,000–500,000).

Various approaches have been attempted for generating such embeddings,
including static algorithms that operate on a corpus of tokenized data
as preprocessors for NLP tasks. Transformers, however, adjust their
embeddings during training.

## Transformer Model Architecture

Now let’s look at the base elements that make up a Transformer by
dissecting the popular GPT2 model

``` python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained('gpt2')
logger.info(model)
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-07-26 </span><span style="color: #bfbfbf; text-decoration-color: #bfbfbf">19:15:02,094361</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_18750</span>/<span style="color: #000080; text-decoration-color: #000080">865827860</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">3</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]<span style="color: #800080; text-decoration-color: #800080; font-weight: bold">GPT2LMHeadModel</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span>                                  
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span>transformer<span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">GPT2Model</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span>                                                                                        
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span>wte<span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Embedding</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span><span style="color: #008080; text-decoration-color: #008080">50257</span>, <span style="color: #008080; text-decoration-color: #008080">768</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>                                                                                   
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span>wpe<span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Embedding</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span><span style="color: #008080; text-decoration-color: #008080">1024</span>, <span style="color: #008080; text-decoration-color: #008080">768</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>                                                                                    
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span>drop<span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Dropout</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span><span style="color: #0000ff; text-decoration-color: #0000ff">p</span>=<span style="color: #008080; text-decoration-color: #008080">0.1</span>, <span style="color: #0000ff; text-decoration-color: #0000ff">inplace</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>                                                                          
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span>h<span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">ModuleList</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span>                                                                                               
      <span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span><span style="color: #008080; text-decoration-color: #008080">0</span>-<span style="color: #008080; text-decoration-color: #008080">11</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>: <span style="color: #008080; text-decoration-color: #008080">12</span> x <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">GPT2Block</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span>                                                                                      
        <span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span>ln_1<span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">LayerNorm</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">((</span><span style="color: #008080; text-decoration-color: #008080">768</span>,<span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>, <span style="color: #0000ff; text-decoration-color: #0000ff">eps</span>=<span style="color: #008080; text-decoration-color: #008080">1e-05</span>, <span style="color: #0000ff; text-decoration-color: #0000ff">elementwise_affine</span>=<span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>                                              
        <span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span>attn<span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">GPT2Attention</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span>                                                                                     
          <span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span>c_attn<span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Conv1D</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span><span style="color: #0000ff; text-decoration-color: #0000ff">nf</span>=<span style="color: #008080; text-decoration-color: #008080">2304</span>, <span style="color: #0000ff; text-decoration-color: #0000ff">nx</span>=<span style="color: #008080; text-decoration-color: #008080">768</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>                                                                        
          <span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span>c_proj<span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Conv1D</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span><span style="color: #0000ff; text-decoration-color: #0000ff">nf</span>=<span style="color: #008080; text-decoration-color: #008080">768</span>, <span style="color: #0000ff; text-decoration-color: #0000ff">nx</span>=<span style="color: #008080; text-decoration-color: #008080">768</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>                                                                         
          <span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span>attn_dropout<span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Dropout</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span><span style="color: #0000ff; text-decoration-color: #0000ff">p</span>=<span style="color: #008080; text-decoration-color: #008080">0.1</span>, <span style="color: #0000ff; text-decoration-color: #0000ff">inplace</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>                                                            
          <span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span>resid_dropout<span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Dropout</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span><span style="color: #0000ff; text-decoration-color: #0000ff">p</span>=<span style="color: #008080; text-decoration-color: #008080">0.1</span>, <span style="color: #0000ff; text-decoration-color: #0000ff">inplace</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>                                                           
        <span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>                                                                                                          
        <span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span>ln_2<span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">LayerNorm</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">((</span><span style="color: #008080; text-decoration-color: #008080">768</span>,<span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>, <span style="color: #0000ff; text-decoration-color: #0000ff">eps</span>=<span style="color: #008080; text-decoration-color: #008080">1e-05</span>, <span style="color: #0000ff; text-decoration-color: #0000ff">elementwise_affine</span>=<span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>                                              
        <span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span>mlp<span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">GPT2MLP</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span>                                                                                            
          <span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span>c_fc<span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Conv1D</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span><span style="color: #0000ff; text-decoration-color: #0000ff">nf</span>=<span style="color: #008080; text-decoration-color: #008080">3072</span>, <span style="color: #0000ff; text-decoration-color: #0000ff">nx</span>=<span style="color: #008080; text-decoration-color: #008080">768</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>                                                                          
          <span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span>c_proj<span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Conv1D</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span><span style="color: #0000ff; text-decoration-color: #0000ff">nf</span>=<span style="color: #008080; text-decoration-color: #008080">768</span>, <span style="color: #0000ff; text-decoration-color: #0000ff">nx</span>=<span style="color: #008080; text-decoration-color: #008080">3072</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>                                                                        
          <span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span>act<span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">NewGELUActivation</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">()</span>                                                                               
          <span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span>dropout<span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Dropout</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span><span style="color: #0000ff; text-decoration-color: #0000ff">p</span>=<span style="color: #008080; text-decoration-color: #008080">0.1</span>, <span style="color: #0000ff; text-decoration-color: #0000ff">inplace</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>                                                                 
        <span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>                                                                                                          
      <span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>                                                                                                            
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>                                                                                                              
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span>ln_f<span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">LayerNorm</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">((</span><span style="color: #008080; text-decoration-color: #008080">768</span>,<span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>, <span style="color: #0000ff; text-decoration-color: #0000ff">eps</span>=<span style="color: #008080; text-decoration-color: #008080">1e-05</span>, <span style="color: #0000ff; text-decoration-color: #0000ff">elementwise_affine</span>=<span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>                                                  
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>                                                                                                                
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span>lm_head<span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Linear</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span><span style="color: #0000ff; text-decoration-color: #0000ff">in_features</span>=<span style="color: #008080; text-decoration-color: #008080">768</span>, <span style="color: #0000ff; text-decoration-color: #0000ff">out_features</span>=<span style="color: #008080; text-decoration-color: #008080">50257</span>, <span style="color: #0000ff; text-decoration-color: #0000ff">bias</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>                                               
<span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>                                                                                                                  
</pre>

GPT2 is an example of a Transformer Decoder which is used to generate
novel text.

Decoder models use only the decoder of a Transformer model. At each
stage, for a given word the attention layers can only access the words
positioned before it in the sentence. These models are often called
auto-regressive models. The pretraining of decoder models usually
revolves around predicting the next word in the sentence.

These models are best suited for tasks involving text generation.

The architecture of GPT-2 is inspired by the paper: “Generating
Wikipedia by Summarizing Long Sequences” which is another arrangement of
the transformer block that can do language modeling. This model threw
away the encoder and thus is known as the “Transformer-Decoder”.

<div>

<img src="https://github.com/architvasan/ai_science_local/blob/main/images/transformer-decoder-intro.png?raw=1" width="500"/>

</div>

[Illustrated GPT2](https://jalammar.github.io/illustrated-gpt2/)

Key components of the transformer architecture include:

- Input Embeddings: Word embedding or word vectors help us represent
  words or text as a numeric vector where words with similar meanings
  have the similar representation.

- Positional Encoding: Injects information about the position of words
  in a sequence, helping the model understand word order.

- Self-Attention Mechanism: Allows the model to weigh the importance of
  different words in a sentence, enabling it to effectively capture
  contextual information.

- Feedforward Neural Networks: Process information from self-attention
  layers to generate output for each word/token.

- Layer Normalization and Residual Connections: Aid in stabilizing
  training and mitigating the vanishing gradient problem.

- Transformer Blocks: Comprised of multiple layers of self-attention and
  feedforward neural networks, stacked together to form the model.

### Attention mechanisms

Since attention mechanisms are arguably the most powerful component of
the Transformer, let’s discuss this in a little more detail.

Suppose the following sentence is an input sentence we want to translate
using an LLM:

`”The animal didn't cross the street because it was too tired”`

To understand a full sentence, the model needs to understand what each
word means in relation to other words.

For example, when we read the sentence:
`”The animal didn't cross the street because it was too tired”` we know
intuitively that the word `"it"` refers to `"animal"`, the state for
`"it"` is `"tired"`, and the associated action is `"didn't cross"`.

However, the model needs a way to learn all of this information in a
simple yet generalizable way. What makes Transformers particularly
powerful compared to earlier sequential architectures is how it encodes
context with the **self-attention mechanism**.

As the model processes each word in the input sequence, attention looks
at other positions in the input sequence for clues to a better
understanding for this word.

<div>

<img src="https://github.com/architvasan/ai_science_local/blob/main/images/transformer_self-attention_visualization.png?raw=1" width="400"/>

</div>

[The Illustrated
Transformer](https://jalammar.github.io/illustrated-transformer/)

#### Multi-head attention

In practice, multiple attention heads are used simultaneously.

This: \* Expands the model’s ability to focus on different positions. \*
Prevents the attention to be dominated by the word itself.

#### Let’s see multi-head attention mechanisms in action!

We are going to use the powerful visualization tool bertviz, which
allows an interactive experience of the attention mechanisms. Normally
these mechanisms are abstracted away but this will allow us to inspect
our model in more detail.

``` python
!pip install bertviz
```

Let’s load in the model, GPT2 and look at the attention mechanisms.

**Hint… click on the different blocks in the visualization to see the
attention**

``` python
from transformers import AutoTokenizer, AutoModel, utils, AutoModelForCausalLM

from bertviz import model_view
utils.logging.set_verbosity_error()  # Suppress standard warnings

model_name = 'openai-community/gpt2'
input_text = "No, I am your father"
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer.encode(input_text, return_tensors='pt')  # Tokenize input text
outputs = model(inputs)  # Run model
attention = outputs[-1]  # Retrieve attention from model outputs
tokens = tokenizer.convert_ids_to_tokens(inputs[0])  # Convert input ids to token strings
model_view(attention, tokens)  # Display model view
```

<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"></script>

      
        <div id="bertviz-e6df04fe758c4b71984d9c45f767ec72" style="font-family:'Helvetica Neue', Helvetica, Arial, sans-serif;">
            <span style="user-select:none">
                &#10;            </span>
            <div id='vis'></div>
        </div>
    &#10;

    <IPython.core.display.Javascript object>

## Pipeline using HuggingFace

Now, let’s see a practical application of LLMs using a HuggingFace
pipeline for classification.

This involves a few steps including: 1. Setting up a prompt 2. Loading
in a pretrained model 3. Loading in the tokenizer and tokenizing input
text 4. Performing model inference 5. Interpreting inference output

``` python
# STEP 0 : Installations and imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch
import torch.nn.functional as F
```

### 1. Setting up a prompt

A “prompt” refers to a specific input or query provided to a language
model. They guide the text processing and generation by providing the
context for the model to generate coherent and relevant text based on
the given input.

The choice and structure of the prompt depends on the specific task, the
context and desired output. Prompts can be “discrete” or “instructive”
where they are explicit instructions or questions directed to the
language model. They can also be more nuanced by more providing
suggestions, directions and contexts to the model.

We will use very simple prompts in this tutorial section, but we will
learn more about prompt engineering and how it helps in optimizing the
performance of the model for a given use case in the following
tutorials.

``` python
# STEP 1 : Set up the prompt
input_text = "The panoramic view of the ocean was breathtaking."
```

### 2. Loading Pretrained Models

The AutoModelForSequenceClassification from_pretrained() method
instantiates a sequence classification model.

Refer to
https://huggingface.co/transformers/v3.0.2/model_doc/auto.html#automodels
for the list of model classes supported.

“from_pretrained” method downloads the pre-trained weights from the
Hugging Face Model Hub or the specified URL if the model is not already
cached locally. It then loads the weights into the instantiated model,
initializing the model parameters with the pre-trained values.

The model cache contains:

- model configuration (config.json)
- pretrained model weights (model.safetensors)
- tokenizer information (tokenizer.json, vocab.json, merges.txt,
  tokenizer.model)

``` python
# STEP 2 : Load the pretrained model.
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
logger.info(config)
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-07-26 </span><span style="color: #bfbfbf; text-decoration-color: #bfbfbf">19:16:13,577577</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_18750</span>/<span style="color: #000080; text-decoration-color: #000080">2259886585</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">5</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]DistilBertConfig <span style="color: #ff00ff; text-decoration-color: #ff00ff">{</span>                               
  <span style="color: #008000; text-decoration-color: #008000">"activation"</span>: <span style="color: #008000; text-decoration-color: #008000">"gelu"</span>,                                                                                            
  <span style="color: #008000; text-decoration-color: #008000">"architectures"</span>: <span style="color: #ff00ff; text-decoration-color: #ff00ff">[</span>                                                                                               
    <span style="color: #008000; text-decoration-color: #008000">"DistilBertForSequenceClassification"</span>                                                                          
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">]</span>,                                                                                                               
  <span style="color: #008000; text-decoration-color: #008000">"attention_dropout"</span>: <span style="color: #008080; text-decoration-color: #008080">0.1</span>,                                                                                        
  <span style="color: #008000; text-decoration-color: #008000">"dim"</span>: <span style="color: #008080; text-decoration-color: #008080">768</span>,                                                                                                      
  <span style="color: #008000; text-decoration-color: #008000">"dropout"</span>: <span style="color: #008080; text-decoration-color: #008080">0.1</span>,                                                                                                  
  <span style="color: #008000; text-decoration-color: #008000">"finetuning_task"</span>: <span style="color: #008000; text-decoration-color: #008000">"sst-2"</span>,                                                                                      
  <span style="color: #008000; text-decoration-color: #008000">"hidden_dim"</span>: <span style="color: #008080; text-decoration-color: #008080">3072</span>,                                                                                              
  <span style="color: #008000; text-decoration-color: #008000">"id2label"</span>: <span style="color: #ff00ff; text-decoration-color: #ff00ff">{</span>                                                                                                    
    <span style="color: #008000; text-decoration-color: #008000">"0"</span>: <span style="color: #008000; text-decoration-color: #008000">"NEGATIVE"</span>,                                                                                               
    <span style="color: #008000; text-decoration-color: #008000">"1"</span>: <span style="color: #008000; text-decoration-color: #008000">"POSITIVE"</span>                                                                                                
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">}</span>,                                                                                                               
  <span style="color: #008000; text-decoration-color: #008000">"initializer_range"</span>: <span style="color: #008080; text-decoration-color: #008080">0.02</span>,                                                                                       
  <span style="color: #008000; text-decoration-color: #008000">"label2id"</span>: <span style="color: #ff00ff; text-decoration-color: #ff00ff">{</span>                                                                                                    
    <span style="color: #008000; text-decoration-color: #008000">"NEGATIVE"</span>: <span style="color: #008080; text-decoration-color: #008080">0</span>,                                                                                                 
    <span style="color: #008000; text-decoration-color: #008000">"POSITIVE"</span>: <span style="color: #008080; text-decoration-color: #008080">1</span>                                                                                                  
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">}</span>,                                                                                                               
  <span style="color: #008000; text-decoration-color: #008000">"max_position_embeddings"</span>: <span style="color: #008080; text-decoration-color: #008080">512</span>,                                                                                  
  <span style="color: #008000; text-decoration-color: #008000">"model_type"</span>: <span style="color: #008000; text-decoration-color: #008000">"distilbert"</span>,                                                                                      
  <span style="color: #008000; text-decoration-color: #008000">"n_heads"</span>: <span style="color: #008080; text-decoration-color: #008080">12</span>,                                                                                                   
  <span style="color: #008000; text-decoration-color: #008000">"n_layers"</span>: <span style="color: #008080; text-decoration-color: #008080">6</span>,                                                                                                   
  <span style="color: #008000; text-decoration-color: #008000">"output_past"</span>: true,                                                                                             
  <span style="color: #008000; text-decoration-color: #008000">"pad_token_id"</span>: <span style="color: #008080; text-decoration-color: #008080">0</span>,                                                                                               
  <span style="color: #008000; text-decoration-color: #008000">"qa_dropout"</span>: <span style="color: #008080; text-decoration-color: #008080">0.1</span>,                                                                                               
  <span style="color: #008000; text-decoration-color: #008000">"seq_classif_dropout"</span>: <span style="color: #008080; text-decoration-color: #008080">0.2</span>,                                                                                      
  <span style="color: #008000; text-decoration-color: #008000">"sinusoidal_pos_embds"</span>: false,                                                                                   
  <span style="color: #008000; text-decoration-color: #008000">"tie_weights_"</span>: true,                                                                                            
  <span style="color: #008000; text-decoration-color: #008000">"transformers_version"</span>: <span style="color: #008000; text-decoration-color: #008000">"4.53.3"</span>,                                                                                
  <span style="color: #008000; text-decoration-color: #008000">"vocab_size"</span>: <span style="color: #008080; text-decoration-color: #008080">30522</span>                                                                                              
<span style="color: #ff00ff; text-decoration-color: #ff00ff">}</span>                                                                                                                  
                                                                                                                   &#10;</pre>

### 3. Loading in the tokenizer and tokenizing input text

Here, we load in a pretrained tokenizer associated with this model.

``` python
#STEP 3 : Load the tokenizer and tokenize the input text
tokenizer  =  AutoTokenizer.from_pretrained(model_name)
input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]
logger.info(input_ids)
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-07-26 </span><span style="color: #bfbfbf; text-decoration-color: #bfbfbf">19:16:14,437902</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_18750</span>/<span style="color: #000080; text-decoration-color: #000080">1325198429</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">4</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]<span style="color: #800080; text-decoration-color: #800080; font-weight: bold">tensor</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">([[</span>  <span style="color: #008080; text-decoration-color: #008080">101</span>,  <span style="color: #008080; text-decoration-color: #008080">1996</span>,  <span style="color: #008080; text-decoration-color: #008080">6090</span>,  <span style="color: #008080; text-decoration-color: #008080">6525</span>,  <span style="color: #008080; text-decoration-color: #008080">7712</span>,      
<span style="color: #008080; text-decoration-color: #008080">3193</span>,  <span style="color: #008080; text-decoration-color: #008080">1997</span>,  <span style="color: #008080; text-decoration-color: #008080">1996</span>,  <span style="color: #008080; text-decoration-color: #008080">4153</span>,  <span style="color: #008080; text-decoration-color: #008080">2001</span>,                                                                                  
          <span style="color: #008080; text-decoration-color: #008080">3052</span>, <span style="color: #008080; text-decoration-color: #008080">17904</span>,  <span style="color: #008080; text-decoration-color: #008080">1012</span>,   <span style="color: #008080; text-decoration-color: #008080">102</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">]])</span>                                                                             
</pre>

### 4. Performing inference and interpreting

Here, we: \* load data into the model, \* perform inference to obtain
logits, \* Convert logits into probabilities \* According to
probabilities assign label

The end result is that we can predict whether the input phrase is
positive or negative.

``` python
# STEP 5 : Perform inference
outputs = model(input_ids)
result = outputs.logits
logger.info(result)

# STEP 6 :  Interpret the output.
probabilities = F.softmax(result, dim=-1)
logger.info(probabilities)
predicted_class = torch.argmax(probabilities, dim=-1).item()
labels = ["NEGATIVE", "POSITIVE"]
out_string = "[{'label': '" + str(labels[predicted_class]) + "', 'score': " + str(probabilities[0][predicted_class].tolist()) + "}]"
logger.info(out_string)
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-07-26 </span><span style="color: #bfbfbf; text-decoration-color: #bfbfbf">19:16:15,627310</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_18750</span>/<span style="color: #000080; text-decoration-color: #000080">4091457049</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">4</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]<span style="color: #800080; text-decoration-color: #800080; font-weight: bold">tensor</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">([[</span><span style="color: #008080; text-decoration-color: #008080">-4.2767</span>,  <span style="color: #008080; text-decoration-color: #008080">4.5486</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">]]</span>,                     
<span style="color: #0000ff; text-decoration-color: #0000ff">grad_fn</span>=<span style="font-weight: bold">&lt;</span><span style="color: #ff00ff; text-decoration-color: #ff00ff; font-weight: bold">AddmmBackward0</span><span style="font-weight: bold">&gt;</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>                                                                                          
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-07-26 </span><span style="color: #bfbfbf; text-decoration-color: #bfbfbf">19:16:15,632337</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_18750</span>/<span style="color: #000080; text-decoration-color: #000080">4091457049</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">8</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]<span style="color: #800080; text-decoration-color: #800080; font-weight: bold">tensor</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">([[</span><span style="color: #008080; text-decoration-color: #008080">1.4695e-04</span>, <span style="color: #008080; text-decoration-color: #008080">9.9985e-01</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">]]</span>,               
<span style="color: #0000ff; text-decoration-color: #0000ff">grad_fn</span>=<span style="font-weight: bold">&lt;</span><span style="color: #ff00ff; text-decoration-color: #ff00ff; font-weight: bold">SoftmaxBackward0</span><span style="font-weight: bold">&gt;</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>                                                                                        
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-07-26 </span><span style="color: #bfbfbf; text-decoration-color: #bfbfbf">19:16:15,634736</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_18750</span>/<span style="color: #000080; text-decoration-color: #000080">4091457049</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">12</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]<span style="color: #ff00ff; text-decoration-color: #ff00ff">[{</span><span style="color: #008000; text-decoration-color: #008000">'label'</span>: <span style="color: #008000; text-decoration-color: #008000">'POSITIVE'</span>, <span style="color: #008000; text-decoration-color: #008000">'score'</span>:                 
<span style="color: #008080; text-decoration-color: #008080">0.9998530149459839</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">}]</span>                                                                                               
</pre>

### Saving and loading models

Model can be saved and loaded to and from a local model directory.

``` python
from transformers import AutoModel, AutoModelForCausalLM

# Instantiate and train or fine-tune a model
model = AutoModelForCausalLM.from_pretrained("bert-base-uncased")

# Train or fine-tune the model...

# Save the model to a local directory
directory = "my_local_model"
model.save_pretrained(directory)

# Load a pre-trained model from a local directory
loaded_model = AutoModel.from_pretrained(directory)
```

## Model Hub

The Model Hub is where the members of the Hugging Face community can
host all of their model checkpoints for simple storage, discovery, and
sharing.

- Download pre-trained models with the huggingface_hub client library,
  with Transformers for fine-tuning.
- Make use of Inference API to use models in production settings.
- You can filter for different models for different tasks, frameworks
  used, datasets used, and many more.
- You can select any model, that will show the model card.
- Model card contains information of the model, including the
  description, usage, limitations etc. Some models also have inference
  API’s that can be used directly.

Model Hub Link : https://huggingface.co/docs/hub/en/models-the-hub

Example of a model card :
https://huggingface.co/bert-base-uncased/tree/main

## Recommended reading

- [“The Illustrated Transformer” by Jay
  Alammar](https://jalammar.github.io/illustrated-transformer/)
- [“Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq
  Models With Attention)” by Jay
  Alammar](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
- [“The Illustrated GPT-2 (Visualizing Transformer Language
  Models)”](https://jalammar.github.io/illustrated-gpt2/)
- [“A gentle introduction to positional
  encoding”](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/)
- [“LLM Tutorial Workshop (Argonne National
  Laboratory)”](https://github.com/brettin/llm_tutorial)
- [“LLM Tutorial Workshop Part 2 (Argonne National
  Laboratory)”](https://github.com/argonne-lcf/llm-workshop)

## Homework

1.  Load in a generative model using the HuggingFace pipeline and
    generate text using a batch of prompts.

- Play with generative parameters such as temperature, max_new_tokens,
  and the model itself and explain the effect on the legibility of the
  model response. Try at least 4 different parameter/model combinations.
- Models that can be used include:
  - `google/gemma-2-2b-it`
  - `microsoft/Phi-3-mini-4k-instruct`
  - `meta-llama/Llama-3.2-1B`
  - Any model from this list: [Text-generation
    models](https://huggingface.co/models?pipeline_tag=text-generation)
  - `gpt2` if having trouble loading these models in
- This guide should help! [Text-generation
  strategies](https://huggingface.co/docs/transformers/en/generation_strategies)

2.  Load in 2 models of different parameter size (e.g. GPT2,
    meta-llama/Llama-2-7b-chat-hf, or distilbert/distilgpt2) and analyze
    the BertViz for each. How does the attention mechanisms change
    depending on model size?
