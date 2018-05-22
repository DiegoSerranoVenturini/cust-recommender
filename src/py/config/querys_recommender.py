query_df_recommender = """
with train_cust_prod as (
    select
  case when cust1 isnull then cust2 else cust1 end as cust,
  case when prod1 isnull then prod2 else prod1 end as prod,
  case when qty isnull then 0 else 1 end as qty,
  case when discount='NULL' or discount isnull then 0 else discount::numeric end as discount
from (
  select
    o.customerid cust1,
    o.productid prod1,
    o.qty qty,
    o.unitdiscountpercent discount,
    sub.customerid cust2,
    sub.productid prod2
  from orders o
    full join
    (
      select
        customerid,
        productid
      from (select customerid
            from customers
            order by random()
            limit 35000) cust
            --limit 100) cust
        cross join (select productid
                    from products
                    order by random()
                    limit 8000) products
      order by random()
      limit 70000
      --limit 70
    ) sub on sub.productid = o.productid and sub.customerid = o.customerid
) sub2
order by qty
),
  recommender_data as (
    select
      t1.cust, t1.prod, t1.qty as target,
      t1.discount,
      c2.sourceid, c2.city, c2.country, c2.state,
      date_part('year', c2.joindate) as year_join,
      case when c2.email isnull then 0 else 1 end as has_email,
      p2.productprice as price
    from train_cust_prod t1
    join customers c2 on t1.cust = c2.customerid
    join products p2 on t1.prod = p2.productid
  )
  select * from recommender_data
;
"""