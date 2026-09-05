# Empirical Analysis for "Predictive Conditional Alphas" (Distaso et al. 2026) {#sec-intro}

This project contains the empirical analysis for the paper "Predictive Conditional Alphas" by Distaso et al. (2026). The relevant version of the paper can be found in [bibliography/distaso2026_JASA.pdf](bibliography/distaso2026_JASA.pdf).

The general steps of the empirical analysis are:

- Download and process factor and stock data.
- Extract high-frequency principal components from the factors.
- Estimate realized betas from the stock returns and the principal components.
- Estimate conditional alphas from the risk-adjusted returns.
- Create portfolios based on the conditional alphas and analyze their performance.

The detailed methodology for each step is described in the associated report chapter or `.ipynb` notebook in [src/](src).

This project has an associated repository and PDF report. Both are explained in this text. Some links point to files in the repository and will not work in the PDF report.

The project runs on Python 3.14 and DuckDB. The [uv](https://docs.astral.sh/uv/) config files allow for running the code with the correct package versions. To generate the report, run `src/quarto_render.ps1` at the project root, or similar commands in Linux or MacOS.


## Repository organization

The repository is organized as follows:

- [data](data): not committed due to size.
    - [data/stocks_raw](data/stocks_raw) and [stocks_1min_returns.parquet](stocks_1min_returns.parquet): raw stocks from the TAQ database plus intermediate transformations, and the stock log-returns at the 1-minute frequency.
    - [data/factors_raw/ff6_1min_returns.csv](data/factors_raw/ff6_1min_returns.csv), [data/factors_1min_returns.csv](data/factors_1min_returns.csv), and [data/factors_5min_returns.csv](data/factors_5min_returns.csv): factor returns at the 1-minute and 5-minute frequencies. The raw data is available at [www.sakethaleti.com/data](https://www.sakethaleti.com/data).
    - [data/pca_1min.csv](data/pca_1min.csv) and [data/pca_5min.csv](data/pca_5min.csv): principal components of the factors at the 1-minute and 5-minute frequencies.
    - [data/betas_1min.csv](data/betas_1min.csv) and [data/betas_5min.csv](data/betas_5min.csv): realized betas of the stocks at the 1-minute and 5-minute frequencies.
    - [data/states_raw](data/states_raw): raw state variables for conditional alphas.
    - [data/alphas/](data/alphas/): conditional alphas named like `alphas_{frequency}_sc{h_t scaling factor}_t{trimming strategy}.parquet`, where the trimming strategy can be none `0` or `ln2` (`h_t^xi / ln(t)^2`).
    - [data/matlab](data/matlab), [data/portfolios](data/portfolios): alphas + stock data (volume, etc.), in the way that matlab code expects it, and that `src/portfolios.ipynb` expects it.
    - [data/old_alphas](data/old_alphas): old, single-factor alphas, for reference.
- [src](src): Most files here are sections of the report.
    - [src/data_factors.ipynb](src/data_factors.ipynb), [src/data_stocks.ipynb](src/data_stocks.ipynb): data checking and processing.
    - [src/pca.ipynb](src/pca.ipynb): high-frequency PCA of the factors estimation.
    - [src/betas.ipynb](src/betas.ipynb), [src/betas_mp.py](src/betas_mp.py): realized betas and risk-adjusted returns estimation.
    - [src/alphas.ipynb](src/alphas.ipynb), [src/alphas_mp.py](src/alphas_mp.py): conditional alphas estimation.
    - [src/portfolios.ipynb](src/portfolios.ipynb): portfolio construction and analysis.
    - [src/parameters.py](src/parameters.py), [src/utils.py](src/utils.py): utility scripts.
- [docs](docs): [the full report](docs/distaso2026_empirical_analysis.pdf), a [summary](docs\summary.md), and other documentation assets.
- Configuration files:
    - Docs and repository: [_quarto.yml](_quarto.yml), [README.md](README.md), [.gitignore](.gitignore)
    - Python: [.python-version](.python-version), [pyproject.toml](pyproject.toml), [uv.lock](uv.lock)
