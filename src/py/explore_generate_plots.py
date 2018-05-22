import seaborn as sns
import matplotlib.pyplot as plt
from DataAssignment.src.py.utils.read_data import *
from sklearn import manifold

# palette definition
agilone_palette = ["#ee503b", "#636360", "#37a75a", "#c597e4", "#e67b31", "#287F93"]
sns.palplot(sns.color_palette(agilone_palette))

# aesthetics
sns.set_palette(agilone_palette)
sns.set_style("ticks")

# product tsne
query = """
select p.productid, p.productprice
from products p
;
"""
df = extract_df_from_query(cur, query, ['category', 'float32'], ['prod', 'price'])
product_embeddings = pd.read_csv(PATH_DATA+'embedding_products.csv')
tsne = manifold.TSNE(n_components=2, n_iter=300)
Y = tsne.fit_transform(product_embeddings.drop(['prod'], axis=1))
df_tsne = pd.DataFrame(np.column_stack((product_embeddings['prod'], Y)), columns=['prod', 'dim1', 'dim2'])
df_tsne['prod'] = df_tsne['prod'].astype('category')
df_tsne = df_tsne.merge(df)

cmap = sns.cubehelix_palette(as_cmap=True)
x, y, z = df_tsne.dim1, df_tsne.dim2, df_tsne.price
fig, ax = plt.subplots(figsize=(11.7, 8.27))
points = ax.scatter(x=x, y=y, c=z, data=df_tsne)
fig.colorbar(points)
fig.savefig(PATH_PLOTS+"tsne_embedding_prod.png")

# customers per source evolution
query = """
select s2.sourcename, date_part('year', joindate), count(customerid)
from customers
  join sources s2 on customers.sourceid = s2.sourceid
group by 1,2
order by 1,2
;
"""
df = extract_df_from_query(cur, query, ['category', 'float32', 'float32'], ['source', 'year', 'num_cust'])
grid = sns.FacetGrid(df, col="source", hue="source", col_wrap=3, size=4)
grid.map(plt.plot, "year", "num_cust", marker="o")
grid.savefig(PATH_PLOTS+"customer_evolution.png")

# lets change the year's type to get a barplot
df = extract_df_from_query(cur, query, ['category', 'category', 'float32'], ['source', 'year', 'num_cust'])
plt.figure(figsize=(11.7, 8.27))
sns.barplot(data=df, x="year", y="num_cust", hue="source")
plt.savefig(PATH_PLOTS+"customer_dist.png")
plt.close()

# order evolution
query = """
-- order evolution
select  s2.sourcename, date_part('year', orderdate), count(*)
from orders
  join customers c2 on orders.customerid = c2.customerid
  join sources s2 on c2.sourceid = s2.sourceid
group by 1,2
order by 1,2
;
"""
df = extract_df_from_query(cur, query, ['category', 'float32', 'float32'], ['source', 'year', 'num_orders'])
grid = sns.FacetGrid(df, col="source", hue="source", col_wrap=3, size=4)
grid.map(plt.plot, "year", "num_orders", marker="o")
grid.savefig(PATH_PLOTS+"order_evolution.png")

# Revenue evolution
query = """
select  s2.sourcename, date_part('year', orderdate),
  round(sum(extrevenue)/1000) kextrevenue_per_year,
  round(sum(p.productprice - p.productcost)/1000)
from orders
  join customers c2 on orders.customerid = c2.customerid
  join sources s2 on c2.sourceid = s2.sourceid
  join products p on orders.productid = p.productid
group by 1,2
order by 1,2
;
"""
df = extract_df_from_query(cur, query,
                           ['category', 'float32', 'float32', 'float32'],
                           ['source', 'year', 'ext_rev', 'gross_rev'])
plt.figure(figsize=(11.7, 8.27))
sns.pointplot(data=df, x="year", y="ext_rev", hue="source", alpha=.2)
plt.savefig(PATH_PLOTS+"ext_rev_evolution.png")
plt.close()

# Marketing investment evolution
query = """
-- source investment evolution
select s2.sourcename, m2.year, round(m2.marketingcost/1000) mtkng_invstm
from marketingcosts m2
  join sources s2 on m2.sourceid = s2.sourceid
order by 1,2
;
"""
df = extract_df_from_query(cur, query, ['category', 'float32', 'float32'], ['source', 'year', 'mtkng_invstm'])
plt.figure(figsize=(11.7, 8.27))
sns.pointplot(data=df, x="year", y="mtkng_invstm", hue="source", alpha=.2)
plt.savefig(PATH_PLOTS+"mtkng_investment_evolution.png")
plt.close()


# $ of revenue per $ invested evolution
query = """
-- source (revenue - investment) evolution
select sub.sourcename, sub.year,
  round(1.0*sub.kextrevenue_per_year / sub2.mktng_cost) dol_rev_per_dol_invest
from
(
select  s2.sourcename, date_part('year', orderdate) as year,
  round(sum(extrevenue)/1000) as kextrevenue_per_year
from orders
  join customers c2 on orders.customerid = c2.customerid
  join sources s2 on c2.sourceid = s2.sourceid
  join products p on orders.productid = p.productid
group by 1,2
) sub
join
(
select s2.sourcename, m2.year as year, m2.marketingcost/1000 as mktng_cost
from marketingcosts m2
  join sources s2 on m2.sourceid = s2.sourceid
) sub2
on sub.sourcename = sub2.sourcename and sub.year = sub2.year
;
"""
df = extract_df_from_query(cur, query,
                           ['category', 'category', 'float32'],
                           ['source', 'year', 'dol_rev_per_dol_invest'])
plt.figure(figsize=(11.7, 8.27))
sns.pointplot(data=df, x="year", y="dol_rev_per_dol_invest", hue="source", alpha=.2)
plt.savefig(PATH_PLOTS+"dollar_revenue_per_dollar_invested.png")
plt.close()


# #customers per $ invested evolution
query = """
-- source (customers - investment) evolution
select sub.sourcename, sub.year,
  round(num_customers / sub2.mktng_cost) cust_per_dol_invest
from
(
select s2.sourcename, date_part('year', joindate) as year, count(customerid) as num_customers
from customers
  join sources s2 on customers.sourceid = s2.sourceid
group by 1,2
) sub
join
(
select s2.sourcename, m2.year as year, m2.marketingcost/1000 as mktng_cost
from marketingcosts m2
  join sources s2 on m2.sourceid = s2.sourceid
) sub2
on sub.sourcename = sub2.sourcename and sub.year = sub2.year
;
"""
df = extract_df_from_query(cur, query,
                           ['category', 'category', 'float32'],
                           ['source', 'year', 'cust_per_dol_invest'])
plt.figure(figsize=(11.7, 8.27))
sns.pointplot(data=df, x="year", y="cust_per_dol_invest", hue="source", alpha=.2)
plt.savefig(PATH_PLOTS+"num_customers_per_dollar_invested.png")
plt.close()

# seasonality study
query = """
select s2.sourcename,
  date_trunc('month', orderdate) ym_order,
  sum(qty)
from orders
  join customers c2 on orders.customerid = c2.customerid
  join sources s2 on c2.sourceid = s2.sourceid
  group by 1,2
  order by 1,2
;
"""
df = extract_df_from_query(cur, query,
                           ['category', 'category', 'float32'],
                           ['source', 'ym', 'orders'])
plt.figure(figsize=(11.7, 8.27))
ax = sns.pointplot(data=df, x="ym", y="orders", hue="source", alpha=.2)
labels = ["%i-%02i" % (df.ym.unique()[i].year, df.ym.unique()[i].month) for i in range(len(df.ym.unique()))]
ax.set_xticklabels(labels=labels, rotation=45, fontsize=7)
plt.savefig(PATH_PLOTS+"orders_per_day.png")
plt.close()

# product_price
query = "select productid, productprice from products"

df = extract_df_from_query(cur, query,
                           ['category', 'float32'],
                           ['product', 'price'])

plt.figure(figsize=(11.7, 8.27))
ax = sns.distplot(df.price, bins=20, kde=False, rug=True)
plt.savefig(PATH_PLOTS+"product_price.png")
plt.close()

# overall revenue
query = """
select
  date_part('year', orderdate),
  round((sum(extrevenue) / 1000000)::numeric, 2)
from orders
group by 1
;"""

df = extract_df_from_query(cur, query,
                           ['category', 'float32'],
                           ['year', 'revenue'])
plt.figure(figsize=(11.7, 8.27))
sns.pointplot(data=df, x="year", y="revenue", alpha=.2)
plt.savefig(PATH_PLOTS+"revenue_evolution.png")
plt.close()

print("product_repetition")
# product repetition
query = """
select
  customerid,
  count(productid)
from orders
group by 1
"""

df = extract_df_from_query(cur, query,
                           ['category', 'float32'],
                           ['year', 'orders'])
df.to_csv(PATH_DATA+"product_repetition.csv")


# product repetition
query = """

select
  customerid,
  count(p.productid),
  round(productprice/100)
from orders
  join products p on orders.productid = p.productid
group by 1,3
"""

df = extract_df_from_query(cur, query,
                           ['category', 'float32', 'category'],
                           ['year', 'orders', 'price'])
df.to_csv(PATH_DATA+"product_repetition_by_price.csv")

# discount evo
query = """
with product_discounts_month as (
    select
      p.productid,
      date_part('month', orderdate) m,
      case when unitdiscountpercent = 'NULL'
        then 0
      else 1 end discount
    from orders
      join products p on orders.productid = p.productid
)

select p.m, count(discount)
from product_discounts_month p
group by 1
order by 1
"""

df = extract_df_from_query(cur, query,
                           ['category', 'int64'],
                           ['month', 'num_discount'])

plt.figure(figsize=(11.7, 8.27))
ax = sns.pointplot(data=df, x="month", y="num_discount", alpha=.2)
labels = ["%02i" % (df.month.unique()[i]) for i in range(len(df.month.unique()))]
ax.set_xticklabels(labels=labels, rotation=45, fontsize=7)
plt.savefig(PATH_PLOTS+"evolution_discounts.png")
plt.close()



