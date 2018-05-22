from DataAssignment.src.py.utils.read_data import *
from DataAssignment.src.py.config.querys_mm import *
import numpy as np
import statsmodels.api as sm

df = extract_df_from_query(cur, query_dataset_rev,
                           ['int64', 'category', 'category', 'float32', 'int64', 'int64', 'int64', 'int64', 'float64'],
                           ['target', 'source', 'month', 'mktngc', 'num_discounts', 'targetMM1', 'targetMM2', 'targetMM3', 'maMM6'])

df['month'] = df['month'].apply(lambda x: 'month_%i' % x)
model_rev = sm.MixedLM.from_formula("target ~ 0 + month + mktngc + maMM6 + num_discounts ", data=df,
                                    re_formula='mktngc', groups=df['source'])
model_rev_fitted = model_rev.fit()

df = extract_df_from_query(cur, query_dataset_orders,
                           ['int64', 'category', 'category', 'float32', 'int64', 'int64', 'int64', 'int64', 'float64'],
                           ['target', 'source', 'month', 'mktngc','num_discounts', 'targetMM1', 'targetMM2', 'targetMM3', 'maMM6'])

df['month'] = df['month'].apply(lambda x: 'month_%i' % x)
model_cust = sm.MixedLM.from_formula("target ~ 0 + month + mktngc + maMM6 + num_discounts", data=df,
                                    re_formula='mktngc', groups=df['source'])

model_cust_fitted = model_cust.fit()

coefs_rev = model_rev_fitted.fe_params
random_effects_rev = model_rev_fitted.random_effects
coefs_cust = model_cust_fitted.fe_params
random_effects_cust = model_cust_fitted.random_effects

results = pd.DataFrame(np.column_stack((
    [random_effects_rev[i]['mktngc']+coefs_rev[coefs_rev.index == 'mktngc'][0] for i in df.source.unique()],
    [random_effects_cust[i]['mktngc']+coefs_cust[coefs_cust.index == 'mktngc'][0] for i in df.source.unique()])),
    columns=['coef_in_rev', 'coef_in_acq', 'source'])

import matplotlib.pyplot as plt
import seaborn as sns


seasonality_coefs = ['month' in i for i in coefs_rev.index.tolist()]
plt.figure(figsize=(12, 6))
ax2 = sns.heatmap(np.transpose(np.matrix(coefs_rev[seasonality_coefs])), annot=True,
            cmap=sns.light_palette("red", as_cmap=True), linewidths=1)
labels = [i+1 for i in range(len(coefs_rev[seasonality_coefs]))]
ax2.set_yticklabels(labels=labels, rotation=0, fontsize=12)
plt.savefig(PATH_PLOTS+"coefs_seasonality.png")
plt.close()

plt.figure(figsize=(12, 6))
ax = plt.subplot(1, 2, 1)
sns.heatmap(np.transpose(np.matrix(results['coef_in_rev'])), annot=True,
            cmap=sns.light_palette("green", as_cmap=True), linewidths=1)
labels = [i for i in df.source.unique()]
ax.set_yticklabels(labels=labels, rotation=0, fontsize=12)
ax.set_xticklabels(labels=['coef_in_rev'], rotation=0, fontsize=12)

ax = plt.subplot(1, 2, 2)
sns.heatmap(np.transpose(np.matrix(results['coef_in_acq'])), annot=True,
            cmap=sns.light_palette("orange", as_cmap=True), linewidths=1)
ax.set_xticklabels(labels=['coef_in_acq'], rotation=0, fontsize=12)
ax.set_yticklabels(labels=[" " for i in df.source.unique()], rotation=0, fontsize=12)
plt.tight_layout(h_pad=5, w_pad=6)

plt.savefig(PATH_PLOTS+"coefs_rev_cust.png")
plt.close()

print('eof')

