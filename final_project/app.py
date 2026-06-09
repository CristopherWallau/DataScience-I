import streamlit as st
import pandas as pd
import altair as alt
import kagglehub

# Comando no terminal para rodar este arquivo: python -m streamlit run app.py
# Comando para acessar o dataset
path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Dashboard Olist", page_icon="📊", layout="wide")

st.title("📦 Análise Estratégica Olist (UFRGS)")
st.markdown("Dashboard interativo cobrindo sazonalidade, comportamento, finanças e logística.")

# --- CARREGAMENTO DE DADOS ---
@st.cache_data
def carregar_dados():
    # Substitua pelo seu CSV final
    df = pd.read_csv(path + "/amostra_olist_consolidada.csv")
    
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    if 'order_delivered_customer_date' in df.columns:
        df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
        df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])
    if 'product_category_name' in df.columns:
        df['product_category_name'] = df['product_category_name'].str.replace('_', ' ').str.title()
    return df

df_final = carregar_dados()

# --- ESTRUTURA DE ABAS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Visão Geral & Produtos", 
    "💳 Pagamentos & Comportamento", 
    "🚚 Logística & Qualidade",
    "🗺️ Análise Geográfica" # Espaço para o seu mapa
])

# ==========================================
# ABA 1: VISÃO GERAL & PRODUTOS
# ==========================================
with tab1:
    st.header("Sazonalidade: Volume vs. Faturamento")
    # 1. Sazonalidade
    df_final['valor_total'] = df_final['price'] + df_final['freight_value']
    sazonalidade_completa = df_final.groupby(df_final['order_purchase_timestamp'].dt.to_period('M')).agg({'order_id': 'nunique', 'valor_total': 'sum'}).reset_index()
    sazonalidade_completa.columns = ['Mes', 'Total_Pedidos', 'Gasto_Total']
    sazonalidade_completa['Mes'] = sazonalidade_completa['Mes'].dt.to_timestamp()

    base = alt.Chart(sazonalidade_completa).encode(x=alt.X('Mes', title='Mês da Compra', axis=alt.Axis(format='%b %Y')))
    line = base.mark_line(color='#1f77b4', size=3).encode(y=alt.Y('Total_Pedidos', title='Total Pedidos'), tooltip=['Mes', 'Total_Pedidos']).properties(height=250)
    bar = base.mark_bar(color='#d62728', size=20).encode(y=alt.Y('Gasto_Total', title='Faturamento (R$)'), tooltip=['Mes', alt.Tooltip('Gasto_Total', format='$,.2f')]).properties(height=150)
    st.altair_chart((line & bar).interactive(), use_container_width=True)

    st.divider()

    # 2. Categorias Mais Vendidas (Matriz)
    st.header("Matriz de Produtos: Categorias Mais Vendidas")
    df_categorias = df_final.groupby('product_category_name').agg({'order_id': 'nunique', 'price': 'sum'}).reset_index()
    top_categorias = df_categorias.nlargest(15, 'order_id')

    scatter = alt.Chart(df_categorias).mark_circle(size=80, opacity=0.7, color='#2ca02c').encode(
        x=alt.X('order_id:Q', title='Volume de Vendas'), y=alt.Y('price:Q', title='Faturamento Total (R$)'),
        tooltip=['product_category_name:N', 'order_id:Q', alt.Tooltip('price:Q', format='$,.2f')]
    ).properties(width=400, height=400).interactive()

    barras_cat = alt.Chart(top_categorias).mark_bar(color='#1f77b4').encode(
        x=alt.X('order_id:Q', title='Volume de Vendas'), y=alt.Y('product_category_name:N', sort='-x', title='Categoria'),
        tooltip=['product_category_name:N', 'order_id:Q']
    ).properties(width=300, height=400)
    
    st.altair_chart((scatter | barras_cat), use_container_width=True)

# ==========================================
# ABA 2: PAGAMENTOS & COMPORTAMENTO
# ==========================================
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        # 3. Preferência de Pagamento
        st.subheader("Distribuição de Pagamentos")
        df_pag = df_final[df_final['payment_type'] != 'not_defined'].copy()
        
        barras_vol = alt.Chart(df_pag).mark_bar().encode(
            x=alt.X('payment_type:N', title='Método', sort='-y'), y=alt.Y('count():Q', title='Volume')
        ).properties(width=300, height=300)
        
        box = alt.Chart(df_pag).mark_boxplot(size=40, clip=True).encode(
            x=alt.X('payment_type:N', title='Método', sort='-y'), y=alt.Y('payment_value:Q', title='Ticket (R$)', scale=alt.Scale(domain=[0, 800]))
        ).properties(width=300, height=300)
        
        st.altair_chart((barras_vol | box), use_container_width=True)
        
    with col2:
        # 4. Relação Parcelamento vs Valor
        st.subheader("Impacto do Parcelamento")
        df_parc = df_final.groupby('payment_installments')['payment_value'].mean().reset_index()
        barras_parc = alt.Chart(df_parc).mark_bar(color='#ff7f0e').encode(
            x=alt.X('payment_installments:O', title='Número de Parcelas', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('payment_value:Q', title='Ticket Médio (R$)'),
            tooltip=['payment_installments:O', alt.Tooltip('payment_value:Q', format='$,.2f')]
        ).properties(height=300)
        st.altair_chart(barras_parc, use_container_width=True)

    st.divider()

    # 5. Horário Nobre
    st.header("O 'Horário Nobre' das Compras")
    df_final['hora_compra'] = df_final['order_purchase_timestamp'].dt.hour
    df_final['nome_dia'] = df_final['order_purchase_timestamp'].dt.dayofweek.map({0: '1-Seg', 1: '2-Ter', 2: '3-Qua', 3: '4-Qui', 4: '5-Sex', 5: '6-Sáb', 6: '7-Dom'})
    df_horarios = df_final.groupby(['nome_dia', 'hora_compra'])['order_id'].nunique().reset_index(name='total_pedidos')

    base_horario = alt.Chart(df_horarios).encode(x=alt.X('nome_dia:N', title='Dia da Semana', sort='ascending'))
    heatmap = base_horario.mark_rect().encode(y=alt.Y('hora_compra:O', title='Hora (0-23h)', sort='descending'), color=alt.Color('total_pedidos:Q', scale=alt.Scale(scheme='blues')), tooltip=['nome_dia:N', 'hora_compra:O', 'total_pedidos:Q']).properties(height=300)
    barras_marginais = base_horario.mark_bar(color='#1f77b4').encode(y=alt.Y('sum(total_pedidos):Q', title='Total Diário')).properties(height=100)
    
    st.altair_chart((heatmap & barras_marginais).interactive(), use_container_width=True)

# ==========================================
# ABA 3: LOGÍSTICA & QUALIDADE
# ==========================================
with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        # 6. Frete vs Peso
        st.subheader("Auditoria: Frete vs. Peso")
        df_frete = df_final.dropna(subset=['product_weight_g', 'freight_value']).copy()
        pontos = alt.Chart(df_frete).mark_circle(opacity=0.4, color='#1f77b4').encode(x='product_weight_g:Q', y='freight_value:Q')
        tendencia = pontos.transform_regression('product_weight_g', 'freight_value').mark_line(color='red')
        st.altair_chart((pontos + tendencia).properties(height=300).interactive(), use_container_width=True)

    with col2:
        # 7. Atraso vs Avaliação
        st.subheader("Custo do Atraso na Avaliação")
        df_an = df_final.dropna(subset=['order_delivered_customer_date', 'order_estimated_delivery_date', 'review_score']).copy()
        df_an['status_entrega'] = (df_an['order_delivered_customer_date'] - df_an['order_estimated_delivery_date']).dt.days.apply(lambda x: 'Atrasado' if x > 0 else 'No Prazo')
        dist_notas = df_an.groupby(['status_entrega', 'review_score']).size().reset_index(name='contagem')
        dist_notas['percentual'] = (dist_notas['contagem'] / dist_notas.groupby('status_entrega')['contagem'].transform('sum')) * 100
        
        graf_atraso = alt.Chart(dist_notas).mark_bar().encode(
            x='status_entrega:N', y='percentual:Q',
            color=alt.Color('review_score:O', scale=alt.Scale(domain=[1,2,3,4,5], range=['#d62728', '#ff7f0e', '#ffbb78', '#98df8a', '#2ca02c']))
        ).properties(height=300)
        st.altair_chart(graf_atraso, use_container_width=True)

    st.divider()
    
    col3, col4 = st.columns(2)
    
    with col3:
        # 8. Curva de Pareto
        st.subheader("Curva de Pareto (Vendedores)")
        df_vend = df_final.groupby('seller_id')['price'].sum().sort_values(ascending=False).reset_index()
        df_vend['porc_receita'] = (df_vend['price'].cumsum() / df_vend['price'].sum()) * 100
        df_vend['porc_vend'] = (pd.Series(range(1, len(df_vend) + 1)) / len(df_vend)) * 100
        
        pareto = alt.Chart(df_vend).mark_area(color='lightblue', line={'color': 'darkblue'}).encode(
            x=alt.X('porc_vend:Q', title='% Vendedores'), y=alt.Y('porc_receita:Q', title='% Receita Acumulada')
        ).properties(height=300)
        st.altair_chart(pareto, use_container_width=True)

    with col4:
        # 9. Melhores vs Piores Categorias
        st.subheader("Extremos de Avaliação")
        df_notas_val = df_final.groupby('product_category_name').agg(nota_media=('review_score', 'mean'), count=('review_score', 'count')).reset_index().query('count >= 5')
        
        base_n = alt.Chart().mark_bar().encode(x=alt.X('nota_media:Q', scale=alt.Scale(domain=[0,5]))).properties(height=250)
        piores = base_n.properties(data=df_notas_val.nsmallest(5, 'nota_media')).encode(y=alt.Y('product_category_name:N', sort='x'), color=alt.ColorValue('#d62728'))
        melhores = base_n.properties(data=df_notas_val.nlargest(5, 'nota_media')).encode(y=alt.Y('product_category_name:N', sort='-x'), color=alt.ColorValue('#2ca02c'))
        
        st.altair_chart((piores | melhores), use_container_width=True)

# ==========================================
# ABA 4: MAPA GEOGRÁFICO
# ==========================================
# ==========================================
# ABA 4: MAPA GEOGRÁFICO
# ==========================================
with tab4:
    st.header("🗺️ Distribuição Geográfica da Demanda")
    st.markdown("Mapeamento de densidade de pedidos e ranking absoluto por estado.")
    
    # 1. Limpeza de anomalias geográficas (mantendo os limites do Brasil)
    df_mapa = df_final.dropna(subset=['cliente_lat', 'cliente_lng'])
    df_mapa = df_mapa[
        (df_mapa['cliente_lat'] < 6) & (df_mapa['cliente_lat'] > -34) &
        (df_mapa['cliente_lng'] < -34) & (df_mapa['cliente_lng'] > -74)
    ]

    # 2. Criando o Mapa de Dispersão (Densidade)
    mapa = alt.Chart(df_mapa).mark_circle(size=15, opacity=0.3, color='#d62728').encode(
        longitude='cliente_lng:Q',
        latitude='cliente_lat:Q',
        tooltip=[
            alt.Tooltip('customer_city:N', title='Cidade'), 
            alt.Tooltip('customer_state:N', title='Estado')
        ]
    ).project(
        type='mercator'
    ).properties(
        title='Densidade Geográfica de Compras',
        height=500
    )

    # 3. Preparando os dados para o Gráfico de Barras
    vendas_estado = df_mapa.groupby('customer_state').size().reset_index(name='Total_Compras')

    # 4. Criando o Gráfico de Barras para quantificar os estados
    barras = alt.Chart(vendas_estado).mark_bar(color='#1f77b4').encode(
        x=alt.X('Total_Compras:Q', title='Volume de Pedidos'),
        y=alt.Y('customer_state:N', sort='-x', title='Estado (UF)'),
        tooltip=[
            alt.Tooltip('customer_state:N', title='Estado'),
            alt.Tooltip('Total_Compras:Q', title='Total de Pedidos')
        ]
    ).properties(
        title='Ranking Absoluto por UF',
        height=500
    )

    # 5. Concatenando e renderizando
    painel_geografico = (mapa | barras).configure_title(fontSize=14, anchor='middle')
    
    st.altair_chart(painel_geografico, use_container_width=True)