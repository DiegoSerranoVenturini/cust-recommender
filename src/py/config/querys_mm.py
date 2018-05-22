query_dataset_rev = """
with sales_per_month_source as (
  select
        s2.sourceid,
        date_trunc('month', orderdate) ym_order,
        sum(extrevenue/1000)                       qty,
        count(case when unitdiscountpercent = 'NULL' then 1 else null end ) num_discounts
      from orders
        join customers c2 on orders.customerid = c2.customerid
        join sources s2 on c2.sourceid = s2.sourceid
      group by 1, 2
),
  moving_avg_mm6 as (
    select t1.sourceid, t1.ym_order, round(avg(t2.qty)::numeric, 2) targetAvgmm6
    from sales_per_month_source t1
      join sales_per_month_source t2
        on t1.ym_order >= (t2.ym_order + interval '6 months') and
           t1.ym_order < (t2.ym_order + interval '13 months') and
           t1.sourceid = t2.sourceid
    group by t1.sourceid, t1.ym_order
    order by 1, 2
  )

select
 auxQ.qty, auxQ.sourcename, auxQ.month, auxQ.mktngC, auxQ.num_discounts, auxQ.qtymm1, auxQ.qtymm2, auxQ.qtymm3, auxQ.targetAvgmm6

from (
  select
    sales_per_month_source.sourceid,
    s2.sourcename,
    sales_per_month_source.ym_order,
    sales_per_month_source.num_discounts,
    qty,
    date_part('month', sales_per_month_source.ym_order) as                       month,
    round((m2.marketingcost / (12 * 1000)) :: numeric, 2) mktngC,
    lag(qty)
    over (
      partition by sales_per_month_source.sourceid
      order by sales_per_month_source.ym_order )        as                       qtymm1,
    lag(qty, 2)
    over (
      partition by sales_per_month_source.sourceid
      order by sales_per_month_source.ym_order )        as                       qtymm2,
    lag(qty, 3)
    over (
      partition by sales_per_month_source.sourceid
      order by sales_per_month_source.ym_order )        as                       qtymm3,
    moving_avg_mm6.targetAvgmm6
  from sales_per_month_source
    join sources s2 on sales_per_month_source.sourceid = s2.sourceid
    join marketingcosts m2
      on sales_per_month_source.sourceid = m2.sourceid
         and date_part('year', sales_per_month_source.ym_order) = m2.year
    join moving_avg_mm6
      on moving_avg_mm6.sourceid = sales_per_month_source.sourceid
         and moving_avg_mm6.ym_order = sales_per_month_source.ym_order
  order by 1, 2, 3
) auxQ;

"""

query_dataset_orders = """
with sales_per_month_source as (
  select
        s2.sourceid,
        date_trunc('month', orderdate) ym_order,
        count(*)                       qty,
        count(case when unitdiscountpercent = 'NULL' then 1 else null end ) num_discounts
      from orders
        join customers c2 on orders.customerid = c2.customerid
        join sources s2 on c2.sourceid = s2.sourceid
      group by 1, 2
),
  moving_avg_mm6 as (
    select t1.sourceid, t1.ym_order, round(avg(t2.qty)::numeric, 2) targetAvgmm6
    from sales_per_month_source t1
      join sales_per_month_source t2
        on t1.ym_order >= (t2.ym_order + interval '6 months') and
           t1.ym_order < (t2.ym_order + interval '13 months') and
           t1.sourceid = t2.sourceid
    group by t1.sourceid, t1.ym_order
    order by 1, 2
  )

select
 auxQ.qty, auxQ.sourcename, auxQ.month, auxQ.mktngC, auxQ.num_discounts, 
 auxQ.qtymm1, auxQ.qtymm2, auxQ.qtymm3, auxQ.targetAvgmm6

from (
  select
    sales_per_month_source.sourceid,
    s2.sourcename,
    sales_per_month_source.ym_order,
    sales_per_month_source.num_discounts,
    qty,
    date_part('month', sales_per_month_source.ym_order) as                       month,
    round((m2.marketingcost / (12 * 1000)) :: numeric, 2) mktngC,
    lag(qty)
    over (
      partition by sales_per_month_source.sourceid
      order by sales_per_month_source.ym_order )        as                       qtymm1,
    lag(qty, 2)
    over (
      partition by sales_per_month_source.sourceid
      order by sales_per_month_source.ym_order )        as                       qtymm2,
    lag(qty, 3)
    over (
      partition by sales_per_month_source.sourceid
      order by sales_per_month_source.ym_order )        as                       qtymm3,
    moving_avg_mm6.targetAvgmm6
  from sales_per_month_source
    join sources s2 on sales_per_month_source.sourceid = s2.sourceid
    join marketingcosts m2
      on sales_per_month_source.sourceid = m2.sourceid
         and date_part('year', sales_per_month_source.ym_order) = m2.year
    join moving_avg_mm6
      on moving_avg_mm6.sourceid = sales_per_month_source.sourceid
         and moving_avg_mm6.ym_order = sales_per_month_source.ym_order
  order by 1, 2, 3
) auxQ
;

"""