# Empirical Analysis for "Predictive conditional alphas" (Ditaso el. al 2026) {#sec-intro}

This project contains the empirical analysis for the paper "Predictive conditional alphas" by Ditaso et al. (2026). The relevant version of the paper can be found [bibliography/Ditaso2026_JASA.pdf](bibliography/Ditaso2026_JASA.pdf).

The general steps of the empirical analysis are:

- Download and process factor and stock data.
- Extract high-frequency principal components from the factors.
- Estimate realized betas from the stock returns and the principal components.
- Estimate conditional alphas from the risk-adjusted returns.

The detailed methodology for each step is described in the associated report chapter or `.ipynb` notebook in [src/](src).

This project has a repository and a PDF report associated. Both are explained in this text. Some links point to files in the repository and won't work in the PDF report.

The project is run in python 3.14 and DuckDB. The [uv](https://docs.astral.sh/uv/) config files allow for running the code with the correct package versions. To generate the report, run `src/quarto_render.ps1` at the project root. Or similar commands in Linux or MacOS.


## Repository organization

The repository is organized as follows:

- [data](data): not committed due to size.
    - [data/stocks_1min_returns](data/stocks_1min_returns) and [data/stocks_5min_returns](data/stocks_5min_returns): individual stock log-returns at the 1-minute and 5-minute frequencies. Stocks are ID'd by their CRSP PERMNO. The raw data [data/trades.db](data/trades.db) is not committed and is from the TAQ database.
    - [data/ff6_1min_returns_raw.csv](data/ff6_1min_returns_raw.csv), [data/ff6_1min_returns.csv](data/ff6_1min_returns.csv) and [data/ff6_5min_returns.csv](data/ff6_5min_returns.csv): Factor returns at the 1-minute and 5-minute frequencies. The raw data is available at [www.sakethaleti.com/data](https://www.sakethaleti.com/data).
- [src](src): Most files here are sections of the report.
    - [src/data_factors.ipynb](src/data_factors.ipynb) [src/data_stocks.ipynb](src/data_stocks.ipynb): Data checking and processing.
    - [src/pca.ipynb](src/pca.ipynb): High-frequency PCA of the factors estimation.
    - [src/betas.ipynb](src/betas.ipynb): Realized betas and risk-adjusted returns estimation.
    - [src/alphas.ipynb](src/alphas.ipynb): Conditional alphas estimation.
    - [src/parameters.py](src/parameters.py), [src/utils.py](src/utils.py): Utility scripts.
- [bibliography](bibliography): reference papers.
- [docs](docs): report of this empirical analysis.
- Configuration files:
    - Docs and repository: [_quarto.yml](_quarto.yml) [README.md](README.md) [.gitignore](.gitignore)
    - Python: [.python-version](.python-version) [pyproject.toml](pyproject.toml)[uv.lock](uv.lock)
