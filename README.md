# cust-recommender

## Project Structure

Inside src/py/ there are tree .py files. One for each part of the analysis.

The recommender can be find in the `core` dir. All its methods are declared in the class `EmbeddingRecommenderProdCust`.

The query to extract the info for training is inside `/config`.

The mixed linear model implemented it is declared and fitted in `main_mixed_linear_model.py`. It uses the library `statsmodels` at its core. There it is possible to find the code to extract the coefficients.

It is necessary though the .py inside `/config` to imports the querys needed for the analysis.

## Recommender

The recommender is build on `pytorch` based on embedding layers just as suggested in the `fast.ai` online MOOC (2018). 

This implementation expands that model by including the possibility of additional (both numerical and categorical with embeddings) features of customers or products. 

Also here the target is binary (if the customer has bought the product or not). So it is more of a propensity model than a classical recommender.
