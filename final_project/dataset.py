import kagglehub
import pandas as pd
import numpy as np

def dataset_download():
    # Download latest version
    path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    return path

def merge_tables(df_list, df_translation=None):
    df_orders, df_customers, df_items, df_payments, df_reviews, df_products, df_sellers, df_geo = df_list

    # 1. Orders Hub
    df_merged = pd.merge(df_orders, df_customers, on='customer_id', how='left')
    df_merged = pd.merge(df_merged, df_payments, on='order_id', how='left')
    df_merged = pd.merge(df_merged, df_reviews, on='order_id', how='left')
    df_merged = pd.merge(df_merged, df_items, on='order_id', how='left')

    # 2. Items Hub
    df_merged = pd.merge(df_merged, df_products, on='product_id', how='left')
    df_merged = pd.merge(df_merged, df_sellers, on='seller_id', how='left')

    # Add translation if provided
    if df_translation is not None:
        df_merged = pd.merge(df_merged, df_translation, on='product_category_name', how='left')

    # 3. Location
    # Both Customers and Sellers connect to olist_geolocation_dataset via zip_code_prefix
    # Deduplicate geo dataset to have one coordinate per zip code
    df_geo_unique = df_geo.drop_duplicates(subset=['geolocation_zip_code_prefix'])
    
    # Customer Location
    geo_customer = df_geo_unique[['geolocation_zip_code_prefix', 'geolocation_lat', 'geolocation_lng']].copy()
    geo_customer.columns = ['customer_zip_code_prefix', 'cliente_lat', 'cliente_lng']
    df_merged = pd.merge(df_merged, geo_customer, on='customer_zip_code_prefix', how='left')

    # Seller Location
    geo_seller = df_geo_unique[['geolocation_zip_code_prefix', 'geolocation_lat', 'geolocation_lng']].copy()
    geo_seller.columns = ['seller_zip_code_prefix', 'seller_lat', 'seller_lng']
    df_merged = pd.merge(df_merged, geo_seller, on='seller_zip_code_prefix', how='left')

    return df_merged

def clean_data(df):
    # 1. Removendo as colunas de texto das avaliações para otimizar espaço
    colunas_para_remover = ['review_comment_title', 'review_comment_message']
    df = df.drop(columns=colunas_para_remover, errors='ignore')

    # 2. Convertendo as colunas de data
    colunas_de_data = [
        'order_purchase_timestamp', 
        'order_approved_at', 
        'order_delivered_carrier_date', 
        'order_delivered_customer_date', 
        'order_estimated_delivery_date',
        'shipping_limit_date',
        'review_creation_date',
        'review_answer_timestamp'
    ]
    for coluna in colunas_de_data:
        if coluna in df.columns:
            df[coluna] = pd.to_datetime(df[coluna], errors='coerce')

    # 3. Limpeza de valores nulos essenciais
    colunas_criticas = [
        'order_id', 
        'customer_id', 
        'product_id', 
        'price',      
        'order_approved_at', 
        'shipping_limit_date', 
        'order_delivered_customer_date', 
        'product_category_name', 
        'review_id', 
        'cliente_lat' 
    ]
    df = df.dropna(subset=colunas_criticas)
    
    # 4. Apenas pedidos entregues
    df = df[df['order_status'] == 'delivered']

    return df

def load_tables(path) -> pd.DataFrame:
    # Configurações de exibição do Pandas
    pd.set_option('display.max_columns', None)

    # Carregando as tabelas principais
    df_orders = pd.read_csv(path + "/olist_orders_dataset.csv")
    df_customers = pd.read_csv(path + "/olist_customers_dataset.csv")
    df_items = pd.read_csv(path + "/olist_order_items_dataset.csv")
    df_payments = pd.read_csv(path + "/olist_order_payments_dataset.csv")
    df_reviews = pd.read_csv(path + "/olist_order_reviews_dataset.csv")
    df_products = pd.read_csv(path + "/olist_products_dataset.csv")
    df_sellers = pd.read_csv(path + "/olist_sellers_dataset.csv")
    df_geo = pd.read_csv(path + "/olist_geolocation_dataset.csv")
    
    try:
        df_translation = pd.read_csv(path + "/product_category_name_translation.csv")
    except FileNotFoundError:
        df_translation = None

    df_list = [df_orders, df_customers, df_items, df_payments, df_reviews, df_products, df_sellers, df_geo]

    df_merged = merge_tables(df_list, df_translation)
    df_cleaned = clean_data(df_merged)
    return df_cleaned

if __name__ == "__main__":
    print("Iniciando download/verificação dos dados via Kaggle...")
    data_path = dataset_download()
    
    print("Carregando, cruzando e limpando as tabelas...")
    df_final = load_tables(data_path)
    
    # Salvando no diretório local do projeto
    output_filename = "amostra_olist_consolidada.csv"
    df_final.to_csv(output_filename, index=False)
    print(f"Dataset final consolidado e salvo com sucesso em '{output_filename}'! (Total: {len(df_final)} linhas)")
