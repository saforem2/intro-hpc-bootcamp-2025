# Intro to HPC Bootcamp 2025
Sam Foreman
2025-07-15

<link rel="preconnect" href="https://fonts.googleapis.com">

- [🐣 Getting Started](#hatching_chick-getting-started)
- [Distributed Training Example](#distributed-training-example)
- [📂 Project Contents](#open_file_folder-project-contents)

## 🐣 Getting Started

1.  Start terminal

2.  Create symlink:

    ``` bash
    # symlink 
    ln -s /global/cfs/cdirs/m4388 $HOME/m4388
    ```

3.  Navigate to `m4388` directory:

    ``` bash
    cd $HOME/m4388
    ```

4.  Clone repo (somewhere) in `$HOME/$USER/`:

    ``` bash
    mkdir $USER && cd $USER
    git clone https://github.com/saforem2/intro-hpc-bootcamp-2025
    ```

5.  Find all Jupyter notebooks:

    ``` bash
    # find all *.ipynb files
    ls **/**/**.ipynb | grep -v "cache" | sort | uniq
    ```

## Distributed Training Example

1.  Login to Perlmutter:

    ``` bash
    ssh <user>@perlmutter.nersc.gov 
    [ -d $HOME/m4388 ] || ln -s /global/cfs/cdirs/m4388 $HOME/m4388
    cd $HOME/m4388/$USER
    git clone https://github.com/saforem2/wordplay
    cd wordplay
    ```

## 📂 Project Contents

- 🏡 [Intro to {AI, HPC} for Science/](./)

  - 📂 [**\[00\] Intro to AI and HPC**/](00-intro-AI-HPC/)
    - 📄 [\[0\] Compute systems](./00-intro-AI-HPC/0-compute-systems/)
    - 📄 [\[1\] Shared-resources](./00-intro-AI-HPC/1-shared-resources/)
    - 📄 [\[2\] Jupyter
      Notebooks](./00-intro-AI-HPC/2-jupyter-notebooks/)
    - 📄 [\[3\] Using Python](./00-intro-AI-HPC/3-python/)
    - 📄 [\[4\] Working with Data](./00-intro-AI-HPC/4-data/)
    - 📗 [\[5\] MCMC Example](./00-intro-AI-HPC/5-mcmc-example/)
    - 📗 [\[6\] Linear
      Regression](./00-intro-AI-HPC/6-linear-regression/)
    - 📗 [\[7\] Statistical
      Learning](./00-intro-AI-HPC/7-statistical-learning/)
    - 📗 [\[8\] Clustering](./00-intro-AI-HPC/8-clustering/)
  - 📂 [**\[01\] Neural Networks/**](./01-neural-networks/)
    - 📄 [\[0\] Intro](./01-neural-networks/0-intro/)
    - 📗 [\[1\] MNIST](./01-neural-networks/1-mnist/)
    - 📗 [\[1\] MNIST (ipynb)](./01-neural-networks/1-mnist-ipynb/)
    - 📗 [\[2\] Advanced](./01-neural-networks/2-advanced/)
    - 📗 [\[3\] Conv. Nets](./01-neural-networks/3-conv-nets/)
    - 📗 [\[4\] Representation
      Learning](./01-neural-networks/4-representation-learning/)
    - 📄 [\[5\] Distributed
      Training](./01-neural-networks/5-distributed-training/)
  - 📂 [**\[02\] Large Language Models**](./02-llms/)
    - 📗 [\[00\] Intro to LLMs](./02-llms/00-intro-to-llms/)
    - 📗 [\[01\] Hands-on LLMs](./02-llms/01-hands-on-llms/)
    - 📄 [\[02\] Prompt Engineering](./02-llms/02-prompt-engineering/)
    - 📗 [\[06\] Parallel Training](./02-llms/06-parallel-training/)
    - 📗 [\[07\] Shakespeare Example](./02-llms/07-shakespeare-example/)
    - 📗 [\[08\] Shakespeare Example
      (colab)](./02-llms/08-shakespeare-example-colab/)

<details class="code-fold">
<summary>👀</summary>

``` python
import datetime
from rich import print
now = datetime.datetime.now()
print(' '.join([ "[#838383]Last Updated[/]:", f"[#E599F7]{now.strftime("%Y-%m-%d")}[/]", "[#838383]@[/]", f"[#00CCFF]{now.strftime("%H:%M:%S")}[/]", ]))
```

</details>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #838383; text-decoration-color: #838383">Last Updated</span>: <span style="color: #e599f7; text-decoration-color: #e599f7; font-weight: bold">2025</span><span style="color: #e599f7; text-decoration-color: #e599f7">-</span><span style="color: #e599f7; text-decoration-color: #e599f7; font-weight: bold">08</span><span style="color: #e599f7; text-decoration-color: #e599f7">-</span><span style="color: #e599f7; text-decoration-color: #e599f7; font-weight: bold">12</span> <span style="color: #838383; text-decoration-color: #838383">@</span> <span style="color: #00ccff; text-decoration-color: #00ccff; font-weight: bold">16:53:50</span>
</pre>
