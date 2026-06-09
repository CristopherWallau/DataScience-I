# Final Project for Data Science I

## Predicting Customer Churn on Olist Dataset

### Classification Model
Uses NLP to:
-> predict which product is being sold.
-> predict if it is a good review or not.

### Regression Model
Predicts the Churn 

### Database Schema Summary

* **Orders Hub:** `olist_orders_dataset` connects to:
    * **Customers** via `customer_id`
    * **Payments**, **Reviews**, and **Order Items** via `order_id`
* **Items Hub:** `olist_order_items_dataset` connects to:
    * **Products** via `product_id`
    * **Sellers** via `seller_id`
* **Location:** Both **Customers** and **Sellers** connect to `olist_geolocation_dataset` via `zip_code_prefix`