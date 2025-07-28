# Intro to HPC Bootcamp 2025
Sam Foreman
2025-07-15

<link rel="preconnect" href="https://fonts.googleapis.com">

- [Contents](#contents)

## Contents

- 🏡 [Intro to {AI, HPC} for Science](./index.qmd)

  - 📂 [00-intro-AI-hpc/](00-intro-AI-HPC/index.qmd)
    - 📄
      [0-compute-systems](./00-intro-AI-hpc/0-compute-systems/index.qmd)
    - 📄
      [1-shared-resources](./00-intro-AI-HPC/1-shared-resources/index.qmd)
    - 📗
      [2-jupyter-notebooks](./00-intro-AI-hpc/2-jupyter-notebooks/index.html)
    - 📄 [3-homework](./00-intro-AI-hpc/3-homework/index.html)
    - 📄 [4-nersc](./00-intro-AI-hpc/4-nersc/index.html)
    - 📗 [5-mcmc-example](./00-intro-AI-hpc/5-mcmc-example/index.html)
    - 📗
      [6-linear-regression](./00-intro-AI-hpc/6-linear-regression/index.html)
  - 📂 [01-neural-networks/](./01-neural-networks/index.html)
    - 📄 [0-intro](./01-neural-networks/0-intro/index.html)
    - 📗 [1-mnist](./01-neural-networks/1-mnist/index.html)
    - 📄 [2-advanced](./01-neural-networks/2-advanced/index.html)
    - 📗 [3-conv-nets](./01-neural-networks/3-conv-nets/index.html)
    - 📗
      [4-representation-learning](./01-neural-networks/4-representation-learning/index.html)
  - 📂 [02-llms/](./02-llms/index.qmd)
    - 📄 [00-intro-to-llms](./02-llms/00-intro-to-llms/index.qmd)
    - 📗 [01-hands-on-llms](./02-llms/01-hands-on-llms/index.ipynb)
    - 📄
      [02-prompt-engineering](./02-llms/02-prompt-engineering/index.qmd)
    - 📗
      [06-parallel-training](./02-llms/06-parallel-training/index.html)
    - 📗
      [07-shakespeare-example](./02-llms/07-shakespeare-example/index.html)
    - 📗
      [08-shakespeare-example-colab](./02-llms/08-shakespeare-example-colab/index.ipynb)
  - 📂 [03-ai-for-science/](./03-ai-for-science/index.html)

``` {pyodide}
#| output_collapsed: true
import ezpz

logger = ezpz.get_logger("root")
logger.info("Welcome to the Intro to HPC Bootcamp 2025!")
logger.info(f"Current time is: {ezpz.get_timestamp()}")
```
