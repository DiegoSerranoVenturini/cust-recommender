import pandas as pd
from sklearn.model_selection import train_test_split
from DataAssignment.src.py.utils.read_data import cur, query_df_recommender, load_recommender_data
from DataAssignment.src.py.utils.utils import create_list_dict
from DataAssignment.src.py.core.recommender import EmbeddingRecommenderProdCust
from DataAssignment.src.py.config.config import PATH_DATA, CATEGORICAL_FEATS, NUMERICAL_FEATS, TARGET
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# recommender data loading
data = load_recommender_data(cur, query_df_recommender, PATH_DATA)

# train target extraction
y_full = data[TARGET]

# train features split
X_full_cat, X_full_num = data[CATEGORICAL_FEATS], data[NUMERICAL_FEATS]

# categorical conversion
categorical_dict = {}
for c in CATEGORICAL_FEATS:
    categorical_dict[c] = create_list_dict(X_full_cat[c].unique().tolist())
    X_full_cat[c] = X_full_cat[c].apply(lambda x: categorical_dict[c][x])
    X_full_cat[c] = X_full_cat[c].astype('category')

# numerical scaling
scaler = StandardScaler()
X_full_num = pd.DataFrame(scaler.fit_transform(X_full_num), columns=NUMERICAL_FEATS)

# train features concatenation
X_full = pd.concat([X_full_cat, X_full_num], axis=1, sort=False)

# categorical seasonality extraction
cat_sz = [(c, len(X_full[c].cat.categories) + 1) for c in CATEGORICAL_FEATS if c not in ['cust', 'prod']]

# categorical embedding dimensionality construction
cat_dim = [(c, min(64, (c + 2) // 2)) for _, c in cat_sz]

# model reporting
print("number of customers = %i || number of products = %i || number of orders for training = %i"
      % (len(categorical_dict['cust']), len(categorical_dict['prod']), data.shape[0]))

print("number of tuples with purchases = %i" % (sum(y_full)))

# train, validation split to test model accuracy
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2)

# categorical index construction
cat_idx = [X_train.columns.get_loc(c) for c in X_train.columns if c in CATEGORICAL_FEATS]

# model initialization
recommender = EmbeddingRecommenderProdCust(
    n_customers=len(categorical_dict['cust'])+1,
    n_products=len(categorical_dict['prod'])+1,
    keys_embedding_dim=64,
    categorical_embedding_dim=cat_dim,
    numerical_dim=len(NUMERICAL_FEATS),
    hidden_dim=[128, 64, 64],
    embedding_drop=0.1,
    hidden_drop=[0.1, 0.2, 0.5]
)

# model training and evaluation
recommender.fit(X_train, y_train, cat_idx, 5, 512, 0.001, (X_test, y_test))
recommender.fit(X_train, y_train, cat_idx, 20, 256, 1e-5, (X_test, y_test))

# product embedding extraction
product_embeddings = recommender.extract_product_embeddings(X_full['prod'].unique())
reverse_prod_dict = dict(map(reversed, categorical_dict['prod'].items()))
product_embeddings['prod'] = product_embeddings['prod'].apply(lambda x : reverse_prod_dict[x])

# saving of the embedding for later analysis
product_embeddings.to_csv(PATH_DATA+'embedding_products.csv', index=False)
# eof
print('training complete')
