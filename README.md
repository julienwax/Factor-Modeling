# Factor modeling project

1. The first step of the project is to sort the S&P500 universe to avoid look-ahead biais and missingness.
Once we have our daily returns we apply statistical technique with time/idio-volatility reweighting on returns to 
apply SVD and create this way a statistical factor model over time.

2. Once our factor model is residualized and factor turnover limited by previous algos, we can start to monitor Information Coefficient and use
linear models to see if we have some signals in analyst forecasts for the residualized returns.
