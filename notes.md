## On how to represent stock time series data to learn HMM

### One HMM representing a stock series.

* The obvious way of representing time series data is by using a table for each stock. Each row corresponds to an observation at a certain time $t$. The columns are attributes measured for that particular stock.
        
* One disadvantage of this representation is that we have to fit one HMM for each stock series.

### One HMM representing a whole market segment.

* In this representation each column contains several attributes from different stocks. So we only need to fit a single HMM representing the joint dynamics of the market or segment overall.

### Fitting models

* Using any of the representations we can fit (using **dynamax** package) an HMM in two ways:
    
  1. Using the raw table shape, that is, with shape $(T, d)$ where $T$ is the total number of observations (total number of time steps) and $d$ es the number of attributes.
        
  2. Determining a time period of length $L < T$ and creating batches.
  

## Some ideas
### Spectral clustering and embedding with hidden Markov models
The paper **2007-Spectral clustering and embedding with hidden Markov models** presents an approach for obtaining a Mercer kernel. We can use this kernel to measure the affinity between two HMMs. It would be interesting if:

* Using said kernel, we apply relational evidential clustering algorithm.

* For each cluster we detect the less affine pair of  stocks.

* Do pairs trading with these stocks.

* Ignoring clusters we could also use the kernel to detect the “globally” less affine pair of stocks and do pairs trading strategies.

### Correlation matrix and pairs trading

* In principle, if $S_1$ and $S_2$ belong to the same cluster, then they have a positive correlation. This is supported by some plots I made in **R**.

* In line with this observation:
    
    * For each cluster we can compute the correlation matrix of price returns (Adj Close price).
    
    * Detect the lowest correlation cofficient in the matrix.
    
    * Create the pair $(S_i, S_j)$ corresponding to this correlation coefficient and do pairs trading.
    
    * I need to think how to use the uncertainty quantification in my favor.