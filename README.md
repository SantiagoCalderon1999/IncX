# IncX

IncX (**Inc**remental E**x**planations) is an advanced method for generating saliency maps and explanations incrementally in real-time.

![Penguin Gif](https://github.com/SantiagoCalderon1999/IncX/blob/main/blob/penguin.gif?raw=true)

## Getting Started

To set up and use this project, you will need `pyenv` for managing Python versions and `poetry` for handling dependencies. Follow these instructions to configure your environment.

### Prerequisites

1. **Install Pyenv**  
   Pyenv allows you to manage multiple Python versions on your system. Select your operating system and follow the relevant installation instructions:

   - **Linux and macOS:** Refer to the guide on the [pyenv GitHub repository](https://github.com/pyenv/pyenv#installation).
   - **Windows:** Use [pyenv-win](https://github.com/pyenv-win/pyenv-win#installation) for installation instructions.

2. **Install the Required Python Version**  
   After installing pyenv, use it to install the Python version specified in the `.python-version` file located in the root directory of this project. Execute the following command:

   ```shell
   pyenv install
   ```

3. **Install Poetry**  
   Install poetry by following the instructions on the [poetry installation page](https://python-poetry.org/docs/#installation).

4. **Configure Poetry**  
   Set up poetry to create virtual environments within the project directory. This configuration helps manage isolated environments for different projects:

   ```shell
   poetry config virtualenvs.in-project true
   ```

5. **Install Project Dependencies**  
   Use poetry to install all required dependencies and create a virtual environment:

   ```shell
   poetry install
   ```

## Running Experiments

To reproduce the experiments comparing D-RISE and IncX, follow these steps using the scripts located in the `/experiments` directory.

1. **Generate Saliency Maps with D-RISE**  
   Navigate to `/experiments/d_rise` and execute the `get_saliency_maps.py` script:

   ```shell
   python get_saliency_maps.py
   ```

   If you are using a high-performance computing (HPC) system, you can run these tasks in parallel by using `batch_experiment_gpu.sh` and `multiple_experiment_gpu.sh`.

2. **Generate Saliency Maps with IncX**  
   Move to `/experiments/incx` and run `get_job_names.py` to create a list of jobs:

   ```shell
   python get_job_names.py
   ```

   Then, execute `get_saliency_maps.py` to compute the saliency maps. For parallel execution on HPC systems, use `batch_experiment_gpu.sh` and `multiple_experiment_gpu.sh`.

3. **Compare Saliency Maps**  
   After generating the saliency maps from both methods, compare them by running:

   ```shell
   python get_blob_names.py
   ```

   Next, obtain the comparison results with:

   ```shell
   python get_similarity_comparison.py
   ```

   On HPC systems, expedite this process with `batch_similarity_comparison.sh` and `multiple_batch_similarity.sh`.

4. **Compute Metrics**  
   To calculate metrics such as insertion, deletion, EPG, and explanation size, start by running:

   ```shell
   python get_blob_names_metrics.py
   ```

   Then, execute:

   ```shell
   python get_metrics.py
   ```

   For HPC systems, speed up this process by using `batch_metrics.sh` and `multiple_batch_metrics.sh`.

## Running Unit Tests

To ensure that the code functions correctly and is thoroughly tested, run:

```shell
poetry run pytest
```

This command will execute all tests in the `tests/` directory, providing feedback on code accuracy and test coverage.

## Linting and Formatting

To maintain high code quality and consistency, use the following commands:

1. **Linting**  
   Run `ruff` to identify and automatically fix code issues:

   ```shell
   poetry run ruff check . --fix
   ```

2. **Formatting**  
   Format your code according to style guidelines with `ruff`:

   ```shell
   poetry run ruff format .
   ```

## Installing the Package

The `Incx` package is available on [PyPI](https://pypi.org/project/incx/). To install the latest version, execute:

```shell
pip install incx
```