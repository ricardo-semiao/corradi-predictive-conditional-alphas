
# Personal understanding of the task

For each asset $i$ and day $t$.

A. Get data:

1. Download the high-frequency data of all considered stocks in the considered window; Download the high-frequency data of all risk factors. The only risk factor considered is the SPDR S&P 500 exchange-traded fund (SPY).
2. Get the state variables $C_t$. These will be provided by you, and are "changes in the VIX index and in the Fed Fund rate, credit and term spreads, momentum, size and value factors, short- and long-term reversal factors, and volatility risk premium".


B. Estimate betas:

1. Find the first timepoint $\tau_{i, t, 1}$ for which there is at least one trade of the asset and at least one observation of all the risk factors. Then, $\tau_{i, t, 2}$, $\tau_{i, t, 3}$, $\dots$.
2. Turn this sequence of $\tau{i, t}$ into a sequence of 1-minute observations, by taking the values of the last observed $\tau$ within the 1-minute interval.
3. Get returns of the asset and factors.
4. Calculate the estimate $\hat{\beta}^{(j)}_{i,t,M{i,t}}$ from appendix B.1, for each factor $j$.

Instead of using only the SPY factor, high-frequency PCA (Ait-Sahalia 2015) could be done to extract statistical factors and their loadings, and used in the alpha estimation step.


C. Estimate alphas:

1. Calculate the adjusted return $\hat{Z}_{i, t+1, M} = r{i, t+1} - \bm{f}'{t+1}\hat{\beta}{i, t, M}$.
2. Run simple PCA on $C_t$ and get all the principal components.
3. Calculate the estimative $\hat{\alpha}_{i,t}$. Plug the data in the codes provided by you.

Run all again with a 5-minute frequency in step B.2.



# Comments of professors

Comments about steps as described by me:
- A1 The task is exactly to consider more factors than simply the SPDR: in practice, take the Fama-French 5-factor model + momentum from https://www.sakethaleti.com/data
- B1+B2 We will make it easier by aggregating to 1- and 5-minute intervals, given that factors are already at the 1-minute. To aggregate, take the last observation before any given time stamp. In other words, forget about refresh time.

Loose steps:
1. homepage on HF returns for each of the risk factors. 5 fama french + momentum
2. agg 1min data into 5min data
3. PCA of the factors every day. every day to be able tu use univariate betas, not multivariate
4. take HF returns of the stocks used (calculate excess return, and then alhas)

loose obs:
- plain realized betas, without demeaning because is HF
- conditioning variables for alpha are relevant
- run routine for every stock
- two methods for alpha, non linear, and (..?)
- data from TAC
- need to check what is 1min and what is 5min
- Aleti describes the way he calculate HF returns. You want to avoid the 9:30 one since it contains overnight returns and we are only considering the intraday part
- middle of july, 7 weeks. maximum a 3week



# Now

What there were:
- a1_Get_betas gets betas from some random TAQ place? While it does interact with apparently stock data, it is not the beta methodology, and the PermRetBetas_CRSP_v07 is not the wanted output
- factor_neutralized_returns_DELIVERY has the bad betas and a repetition of ff6_5min_aggregated, and can be ignored
- it is not clear how we should get stock data, if is from zero, or if the Stocks2_v2 folder can be used
- ff6/Python has the raw 1min FF6 data, and the code to aggregate it to 5min, and the daily PCA, but not to the betas of the components.
- the imperial_returns_betas has 1min and 5min data divided by stock, not clear how they relate to the data of the previous point. This seems to be a AI prompt that gives the data and asks for the calculation of betas

Thus:
- Download the data from https://www.sakethaleti.com/data, and/or from the link of the paper
- Compare with the data in ff6/ folder. Decide what to use
- Re-do the aggregation if needed (and any other preprocessing needed)
- Find about the stock data: it imperial_returns_betas/\*min_returns/? is it trades.db? Decide which to use of find if we need to get it from scratch (then do it)
- Get 1min/5min stock returns, and calculate the betas (simple regression? other?)
- Get the factor-neutralized returns $Z$

Keep going:
- ff6_1min is from https://www.sakethaleti.com/data. it is probably the ff6 data we need, already done, i guess no need to look for the factors in data/proc/...
- lets recreate a better looking code anyways, but following the same steps
- say we redid the pca, got the same results, send it to them. talk about stock data




## Output


## Output

### Old RA Folder Structure Analysis

Based on the files from the previous research assistant, the `old_ra` folder contains code and explanations primarily focused on calculating realized betas and factor-neutralized returns using High-Frequency (HF) Fama-French 6-factor data and PCA.

#### 1. `imperial_returns_betas`
This folder contains the core conceptual documentation (`README.md` and `EXPLAINER.md`) for the pipeline. It maps directly to **Step B (Estimate betas)** and **Step C1 (Calculate adjusted returns)** of your task.
- **Workflow described:**
  1. Computes 1-minute log returns from trade prices, dropping the 9:30 AM overnight return (matching your loose observation).
  2. Aggregates these into 5-minute blocks (78 blocks per day).
  3. Computes daily open-to-close returns.
  4. Runs PCA on the day's 5-minute Fama-French factor returns (this relates to your loose step 3: "PCA of the factors every day... to use univariate betas").
  5. Applies Mancini jump truncation (optional) to drop large jumps to estimate continuous beta.
  6. Computes realized univariate betas on the principal components.
  7. Calculates the factor-neutralized return $Z$ (which corresponds to your $\hat{Z}_{i, t+1, M}$ in Step C1).

#### 2. `ff6` (Python and Matlab)
These folders contain equivalent scripts (`process_ff6_pca.py` and `ff6_pca_analysis.m`) that execute the factor processing and PCA steps outlined in the explainer.
- **Input:** `ff6_1min_returns.csv` (1-min Fama-French 5-factor + momentum).
- **Process:** Converts simple returns to log returns, drops 9:30 AM rows, aggregates into 5-minute blocks, and runs daily PCA for three different configurations (3 factors, 4 factors, 6 factors).
- **Output:** `ff6_5min_aggregated.csv` and PCA score matrices (e.g., `pca_all6.csv`).
- **Relation to task:** This almost completely handles the factor side of **Step A1**, loose step 2 (agg 1min data into 5min data), and loose step 3 (PCA of factors every day).

#### 3. `a1_Get_betas_v05.m`
This appears to be an older script for extracting and merging daily CRSP data with intraday TAQ data from a folder named `Stocks2_v2`.
- **Input:** `Stocks2_v2/PermNo_*.mat` (Stock-specific data containing CRSP daily variables and TAQ intraday observations).
- **Process:** Iterates through stocks, matches dates, and extracts daily open/close prices, volumes, and pre-calculated realized variance/betas (RV 5 min).
- **Output:** `InPutData/PermRetBetas_CRSP_v07` containing the merged panel.
- **Relation to task:** This relates to **Step A1** (High-frequency stock data). However, it seems to rely on pre-processed TAQ data rather than raw tick data, and might not use the PCA factor methodology described in the `imperial_returns_betas` folder. 

#### Recommendations to hit the ground running:
1. **Factor Pipeline:** The Fama-French factor processing is practically done. Choose either the Python or MATLAB version in `ff6/` and ensure you have `ff6_1min_returns.csv`. Running this will give you the daily PC scores.
2. **Stock Data Pipeline:** You need to bridge the gap between `a1_Get_betas_v05.m` and the methodology in `EXPLAINER.md`. Check the structure of `Stocks2_v2/PermNo_*.mat` to see if it contains 1-min/5-min returns. If it does, you can compute the realized betas by regressing these returns against the PC scores obtained in step 1. If it doesn't, you will need to build the stock 5-min return series from raw TAQ tick data.
3. **Compute Neutralized Returns ($Z$):** Implement Step 7 from the `EXPLAINER.md` to compute $Z$, as this acts as the dependent variable for your alpha estimation.
4. **Alpha Estimation:** The previous RA's work ends at the calculation of neutralized returns. You will need to write new code for **Step C2 and C3**: getting the state variables $C_t$, running PCA on them, and plugging the data into the non-parametric kernel estimator codes provided by your professors to find $\hat{\alpha}_{i,t}$.

## More on the data

### Analysis of `a1_Get_betas_v05.m` and Stock `.mat` files

The script `a1_Get_betas_v05.m` is an **ETL (Extract, Transform, Load)** utility. It does not perform any high-frequency financial econometrics itself; instead, it extracts daily metrics and pre-computed measures from a heavily nested directory of `.mat` files and flattens them into a usable panel dataset.

#### 1. The Original Input Data (`Stocks2_v2/PermNo_*.mat`)
These files contain a very convoluted, pre-processed hierarchical structure (a cell array in MATLAB, which translates to lists of arrays/dicts in Python). 
* **What it is:** A combination of daily lower-frequency CRSP data and daily aggregations of high-frequency TAQ data.
* **The Metadata:** The Python dictionary snippets you provided show the file's metadata: it contains the start and end dates (`From: 19930104`, `To: 20171229`), the ticker (`PLXS`), and the company name (`PLEXUS CORP`). 
* **The Data:** The core data is buried in sub-cells: `permInfo{2,1}` holds the CRSP daily data, and `permInfo{2,4}` holds TAQ high-frequency daily summaries.

#### 2. What the Script Does
The script iterates over every stock file and does the following:
1. **Extracts CRSP data:** It grabs 10 specific columns from the CRSP table (Date, Open Price, Close Price, Return, Volume, Market Cap, Number of trades, etc.).
2. **Extracts TAQ data:** It iterates over the TAQ data to extract specific daily boundaries and metrics: total volume, total trades, first trade time/price (open), and last trade time/price (close).
3. **Extracts Pre-computed Betas/Variances:** It pulls pre-calculated "Realized Beta: RV 5 min" and "RV 5 min subsampled" directly from the 10th and 11th indices of the TAQ cell array. **It does not calculate them.**
4. **Merges and Cleans:** It merges these on the `date` key, drops duplicate dates, and drops days with missing data.

#### 3. The Output Data (`PermRetBetas_CRSP_v07.mat`)
The script outputs a simplified list (or MATLAB cell array) where each element represents one stock. The output arrays you saw map to the following:

* **[0] `array(10032.)`**: The **PERMNO** (CRSP's unique permanent number for the stock). Unit: Integer ID.
* **[1] `'PLXS'`**: The **Ticker** symbol. Unit: String.
* **[2] `array(432412.85...)`**: The **Mean of a CRSP variable**. The code takes `mean(BetaData(:,8))`. Based on standard CRSP ordering, this is likely the stock's average historical daily volume or market cap across the entire sample.
* **[3] `array([[...]], shape=(4276, 18))`**: The **Merged Data Panel**. This is a daily time-series matrix of 4,276 days and 18 columns. The columns are:
  * **Cols 1-10:** CRSP data (`date`, `openprc`, `prc`, `ret`, `vol`, etc.)
  * **Cols 11-12:** TAQ daily aggregates (`numtrd2`, `vol2`)
  * **Cols 13-14:** TAQ first trade of the day (`t.open`, `p.open`)
  * **Cols 15-16:** TAQ last trade of the day (`t.close`, `p.close`)
  * **Cols 17-18:** Pre-calculated Realized Variance/Beta (`Bss5m_OC`, `Bss5mCC`)
* **[4] `array(45203020.)`**: A **Metadata Code**. Extracted from deep within the file (`permInfo{secPlc2,6}`). Based on its format, this is almost certainly a historical SIC (Standard Industrial Classification) code or NAICS code.

#### Implications for Your Task
This script is highly likely a **legacy file** from a different or earlier phase of the previous RA's project. 
1. The `EXPLAINER.md` states that factor-neutralized returns ($Z$) are computed by estimating betas against the *Principal Components of the Fama-French 6 factors*.
2. The `a1_Get_betas_v05.m` output only contains two pre-calculated variance/beta columns (`Bss5m_OC`, `Bss5mCC`), not the 6 individual PC betas required for the neutralization step described in the explainer.
3. Therefore, unless there's another script that uses the 18-column matrix to calculate the PC betas, the `Stocks2_v2` dataset might not contain the raw 1-min/5-min intraday log returns needed to run the regressions against the factor PCA scores. You will likely need to find the raw intraday return `.csv` files mentioned in `EXPLAINER.md` (e.g., `5min_returns/perm{N}.csv`), or write a script to generate them.

