---
bibliography: ../docs/references.bib

format:
  pdf:
    output-file: distaso2026_summary
    title-meta: Distaso 2026 Summary
    author-meta: Ricardo Semião e Castro
    documentclass: scrartcl
    classoption: [headings=small, titlepage=false]
    number-depth: 3
    number-sections: true
    include-in-header:  
      text: |
        \usepackage[a4paper, left=2cm, right=2cm, top=2cm, bottom=2.5cm]{geometry}

        \usepackage{amsmath}
        \usepackage{amssymb}
        \usepackage{mathtools}

        \setuptoc{toc}{leveldown}
---

```{=tex}
\begin{center}
\large
\textbf{"Predictive Conditional Alphas" - Methodology Summary}
\end{center}
```

This is a summary of the methodology used in the empirical analysis for the paper.

# Data

## Time window and trading days

The sample period is from 2001-01-02 to 2017-12-29. All the datasets are cut to this window. The first 500 days are used as a warm-up period for the alphas estimation.

Only trading days are considered. Trading days are defined by the NYSE calendar, removing 52 days with less than 100% market-wide factor availability, and other 52 outlier days where the median number of minutes with trades (across all stocks) was 0.1 points smaller than its 5-day rolling mean (and often below the 50% mark).


## Stock data

The data source is intraday tick data from the [NYSE TAQ database](https://www.nyse.com/data-products/catalog/daily-taq), with documentation in [Daily_TAQ_Client_Spec_v4.3.pdf](https://www.nyse.com/publicdocs/nyse/data/Daily_TAQ_Client_Spec_v4.3.pdf).

The database is filtered to include only: NYSE, NYSE-Arca, NASDAQ, and AMEX exchanges; regular trades; days in the sample period; regular trading hours.

Stocks are tracked by their CRSP _PERMNO_ identifier. For each PERMNO, the data is aggregated:

1. For each 1-minute block $m$: get the last seen price before the end of $m$'s block, $p_m$.
2. For each day (i.e. excluding overnight returns), calculate $r^1_m = \ln(p_m / p_{m-1})$.

Minutes without data will be forward filled with $0$ return at the betas calculation stage, and aggregated into 5-minute returns simultaneously.


### Liquidity filter

The quality of alpha estimates for a given day depends on the amount of minutes with data (i.e., minutes where trades happened). Thus, we require stocks to have at most $2.5\%$ of days in its life -- from the first day it was traded to the last, within our sample window -- with less than $78$ minutes with trades ($20\%$ of the trading day). This guarantees that at least $20\%$ of the 5 minute blocks are also available.

Note that requiring $78$ _minutes with trades_ is much more restrictive than $78$ _trades_, as more than one trade can happen in the same minute. Additionally, as the alphas estimation requires at least 500 days for warm-up, stocks with less than 500 days within the sample period are removed. At the end, there are 1141 stocks considered liquid enough.


## Factor data

The factor data is composed of the Fama-French 6 factors from [sakethaleti.com/data](https://www.sakethaleti.com/data), or [this Google Drive link](https://drive.google.com/file/d/1TJqtc-8KlwTmtS6V1rGn8xCj2I0mUdo_). Documentation can be found in Section 1 of the [online appendix](https://drive.google.com/file/d/1vN5jPnuwlZhwb3NZGGHdkMYSePk5Sp45/view). The TAQ database is filtered to include only: Days in the sample period; Regular trading hours; No overnight returns, i.e. excluding minute 9:30. The original data is in returns, and is transformed into log-returns.


# High-frequency PCA of factors

We follow the high-frequency PCA described in  Aït-Sahalia and Xiu (2015) "Principal Component Analysis of High Frequency Data". We define $X$ as the $d = 6$ dimensional time series of the 6 factors log-prices. We have data from $[0, t]$ in increments of $\Delta_n$ (minutes). Our goal is to find instantaneous linear combinations of $X$, orthogonal to each other, that maximize the continuous part of the quadratic variation of $X$.


## Spot covariance matrix

The first step is estimating a spot covariance matrix and its associated eigenvalues. To capture non-stationary dynamics, the time window is divided into non-overlapping blocks of length $k_n \Delta_n$. Within each small block, the continuous covariance $c_s$ is assumed to be roughly constant. At each block $i$, we have $k_n$ log-returns observations (indexed by $j$):

$$
\Delta_{ik_n+j}^n X \coloneqq X_{(ik_n+j) \Delta_n} - X_{(ik_n+j-1)\Delta_n}.
$$

We estimate $c_{i k_n \Delta_n}$ using the block spot covariance estimator of such returns, defined as:

$$
    \hat{c}_{i k_n \Delta_n} = \frac{1}{k_n\,\Delta_n} \sum_{j=1}^{k_n} \left(\Delta_{ik_n+j}^{n}X\right)\left(\Delta_{ik_n+j}^{n}X\right)^{T} \mathbf{1}\left\{\left\|\Delta_{ik_n+j}^{n}\right\|\le u_n\right\},
$$

where $u_n$ is a jump truncation threshold. The $k$ and $u$ parameters can be chosen with the shorthands proposed by the literature. $k_n$ was calculated as $858$, approximately 2 trading days, and the $u_{n}$ values basically removed returns above the $0.995$ percentile for each given factor.

<!-- [^shorthands]: **Block Size ($k_n$):** the divisor of $[t / \Delta_n]$ closest to $\theta \Delta_n^{-1/2} \sqrt{\log(d)}$, with $\theta = 0.5$. **Jump Truncation ($u_{in}$):** Set asset-by-asset as $u_{n} = 3\left(\int_{0}^{t}c_{ii,s}ds/t\right)^{0.5}\Delta^{0.47}_{n}$. Where the internal integral can be estimated with Bipower Variation. -->


## Principal components

The instantaneous eigenvalues $\hat{\lambda}_{i k_n \Delta_n} = \lambda(\hat{c}_{i k_n \Delta_n})$ are obtained from the standard determinant equation: $\operatorname{det}(\hat{c}_{i k_n \Delta_n} - \hat{\lambda}_{i k_n \Delta_n} I) = 0$. From it, we can obtain the standard eigenvectors $\hat{\gamma}_{g, i k_n \Delta_n}$.

For each block $i$, we can retrieve the $k_n$ observations of the $g$-th principal component:

$$
    \widehat{PC}_{g,i} = \Delta X_{ik_n\Delta_n ~:~ (i+1)k_n\Delta_n - 1} \cdot \hat{\gamma}_{g,(i-1)k_n\Delta_n} ~~ \forall i \in \{1, \dots, [t/(k_n\Delta_n)]-1\}
$$

Joining all blocks and PCs generates the $t \times d$ matrix of principal components. Note that PCs use the eigenvectors estimated from the immediately preceding block to project the current block's returns. The first block's eigenvectors are estimated from data preceding the sample window.


# Betas

For each stock $i$ and day $t$, we use the $M$ minute (or 5-minute) observations of the stock log-returns $\Delta_\ell P$ and factor PCs $\Delta_\ell F$ to calculate the realized betas:

$$
    \hat{\beta}_{i,t,M} =
        \left[\sum_{\ell=0}^{M-1} \Delta_\ell F_{t} ~ \Delta_\ell F_{t}^{\prime}\right]^{-1}
        \sum_{\ell=0}^{M-1} \Delta_\ell F_{t} ~ \Delta_\ell P_{i, t}
$$

The alphas will be calculated based on the risk-adjusted returns, which use the betas and the daily log-returns of the factors ($f_t$) and stocks ($r_{i,t}$):

$$
    \hat{Z}_{i,t+1,M} = r_{i,t+1} - f_{t+1}\hat{\beta}_{i,t,M}
$$

The calculations are independent across stocks and parallelized in an outer loop over $i$. Inside, calculations are run for each $\lambda$, operating over the tensor $(d, t, m)$.

Minutes without data are forward filled as $0$ return, and the 5-minute frequency data is created by summing all the 5 minutes within each 5-minute block of returns.


# Alphas

## Methodology

Let then $\widehat{PC}_t = (\widehat{PC}_{1,t}, \dots, \widehat{PC}_{k,t})'$ denote the $k = 3$ principal component estimates of $C_t$. Then, the conditional alpha estimator is:

$$
\hat{\alpha}_{i,t} = \hat{m}_{i,T,M}(\widehat{PC}_t) = \frac{\frac{1}{T h_T^k} \sum_{\ell=1}^{T-1} \hat{Z}_{i,\ell+1,M} K\left(\frac{\widehat{PC}_\ell - \widehat{PC}_t}{h_T}\right)}{\hat{g}_T(\widehat{PC}_t)},
$$

where $\hat{g}_T(\widehat{PC}_t) = \frac{1}{T h_T^k} \sum_{\ell=1}^{T-1} K\left(\frac{\widehat{PC}_\ell - \widehat{PC}_t}{h_T}\right)$ is the kernel estimator of the density $g$ of the vector of principal components. We trim large alphas via $1\{pc \in \hat{G}_T(pc)\}$, where $\hat{G}_T(pc) = \{c : \hat{g}_T(pc) > d_T\}$. We consider $d_T = O(h_T^\zeta)$ with $0 < \zeta < 1/4$.

<!-- The kernel used is Gaussian:

$$
K(u) = \left( \frac{1}{\sqrt{2\pi}} \right)^k \prod_{j=1}^{k} \exp\left( -\frac{1}{2} u_j^2 \right) = \left( \frac{1}{\sqrt{2\pi}} \right)^k \exp\left( -\frac{1}{2} \sum_{j=1}^{k} u_j^2 \right)
$$ -->


## Calculation

The paper proposes an expanding window to restrict attention to real-time information, including the PCs, bandwidth, and trimming threshold calculation. Calculating the PC and kernel bandwidths and trimming thresholds is done separately for each stock, considering only the state variable periods that match the stock's trading days.


The main algorithm sets up the stock and state variables data and runs the procedure for each stock at both 1-minute and 5-minute frequencies. The operations are independent across stocks, so the main loop is parallelized over stocks.
