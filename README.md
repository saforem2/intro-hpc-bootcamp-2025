# Intro to HPC Bootcamp 2025

## Instructions for Building Site

1. [Install Quarto](https://quarto.org/docs/download/)
2. Clone the repository:

   ```bash
   git clone --filter=tree:0 https://github.com/saforem2/intro-hpc-bootcamp-2025
   cd intro-hpc-bootcamp-2025 
   ```

3. Create a virtual environment:

   (see [uv](https://docs.astral.sh/uv/) for details on `uv`)

   ```bash
   uv venv --python=3.13
   source .venv/bin/activate
   uv pip install jupyter-cache ipykernel matplotlib torch torchvision torchdata torchinfo rich ptpython euporie deepspeed mpi4py bertviz "git+https://github.com/saforem2/ezpz[dev]" "git+https://github.com/saforem2/wordplay"
   python3 -m wordplay.prepare
   python3 -m ipykernel install --sys-prefix
   ```

4. Install `mcanouil/quarto-iconify`:

    ```bash
    quarto add mcanouil/quarto-iconify
    ```


5. Build the site:

   ```bash
   quarto preview content
   ```
