import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Supply Chain Analytics", layout="wide")

@st.cache_data
def load_data():
    np.random.seed(42)
    n = 10000
    df = pd.DataFrame({
        'order_id': range(1, n+1),
        'date': pd.date_range(start='2022-01-01', periods=n, freq='D'),
        'product_category': np.random.choice(['Electronics','Clothing','Food','Tools','Medical'], n),
        'warehouse_region': np.random.choice(['North','South','East','West'], n),
        'units_ordered': np.random.randint(1, 500, n),
        'units_in_stock': np.random.randint(0, 1000, n),
        'lead_time_days': np.random.randint(1, 30, n),
        'shipping_cost': np.round(np.random.uniform(5, 500, n), 2),
        'on_time_delivery': np.random.choice([0, 1], n, p=[0.2, 0.8])
    })
    return df

@st.cache_resource
def train_model(df):
    df_ml = df.copy()
    df_ml['month']         = df_ml['date'].dt.month
    df_ml['dayofweek']     = df_ml['date'].dt.dayofweek
    df_ml['quarter']       = df_ml['date'].dt.quarter
    df_ml['dayofyear']     = df_ml['date'].dt.dayofyear
    df_ml['category_code'] = df_ml['product_category'].astype('category').cat.codes
    df_ml['region_code']   = df_ml['warehouse_region'].astype('category').cat.codes
    features = ['month','dayofweek','quarter','dayofyear',
                'category_code','region_code','lead_time_days','units_in_stock','shipping_cost']
    X = df_ml[features]
    y = df_ml['units_ordered']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, rf.predict(X_test))
    return rf, features, mae

df = load_data()
rf, features, mae = train_model(df)

st.sidebar.title("Filters")
selected_categories = st.sidebar.multiselect("Product Category", options=df['product_category'].unique(), default=df['product_category'].unique())
selected_regions = st.sidebar.multiselect("Warehouse Region", options=df['warehouse_region'].unique(), default=df['warehouse_region'].unique())

filtered = df[df['product_category'].isin(selected_categories) & df['warehouse_region'].isin(selected_regions)]

st.title("ML-Driven Supply Chain Analytics Dashboard")
st.markdown("Interactive dashboard for inventory forecasting, demand analysis, and logistics insights.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Orders",   f"{len(filtered):,}")
k2.metric("Avg Lead Time",  f"{filtered['lead_time_days'].mean():.1f} days")
k3.metric("On-Time Rate",   f"{filtered['on_time_delivery'].mean()*100:.1f}%")
k4.metric("Forecast MAE",   f"{mae:.0f} units")

st.divider()

col1, col2 = st.columns(2)
with col1:
    monthly = filtered.groupby(filtered['date'].dt.to_period('M'))['units_ordered'].sum().reset_index()
    monthly['date'] = monthly['date'].astype(str)
    fig = px.line(monthly, x='date', y='units_ordered', title='Monthly Demand Trend', template='plotly_dark')
    fig.update_traces(line_color='#00d4ff')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    delivery = filtered.groupby('warehouse_region').apply(lambda x: round(x['on_time_delivery'].mean()*100,1)).reset_index(name='on_time_rate')
    fig2 = px.bar(delivery, x='warehouse_region', y='on_time_rate', title='On-Time Delivery by Region',
                  color='on_time_rate', color_continuous_scale='RdYlGn', range_y=[60,90], template='plotly_dark')
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    stockout = filtered[filtered['units_ordered'] > filtered['units_in_stock']]
    pivot = stockout.groupby(['product_category','warehouse_region']).size().unstack(fill_value=0)
    fig3 = px.imshow(pivot, title='Stockout Risk Heatmap', color_continuous_scale='YlOrRd', template='plotly_dark', text_auto=True)
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    imp_df = pd.DataFrame({'feature': features, 'importance': rf.feature_importances_}).sort_values('importance')
    fig4 = px.bar(imp_df, x='importance', y='feature', orientation='h', title='🌲 ML Feature Importance',
                  color='importance', color_continuous_scale='Blues', template='plotly_dark')
    st.plotly_chart(fig4, use_container_width=True)

st.divider()
st.subheader("Demand Forecaster")
f1, f2, f3, f4 = st.columns(4)
cat    = f1.selectbox("Category", sorted(df['product_category'].unique()))
region = f2.selectbox("Region",   sorted(df['warehouse_region'].unique()))
stock  = f3.slider("Units in Stock", 0, 1000, 500)
lead   = f4.slider("Lead Time (days)", 1, 30, 15)

cat_codes    = {c: i for i, c in enumerate(sorted(df['product_category'].unique()))}
region_codes = {r: i for i, r in enumerate(sorted(df['warehouse_region'].unique()))}

input_data = pd.DataFrame([{
    'month': 6, 'dayofweek': 2, 'quarter': 2, 'dayofyear': 180,
    'category_code': cat_codes[cat], 'region_code': region_codes[region],
    'lead_time_days': lead, 'units_in_stock': stock, 'shipping_cost': 250.0
}])
prediction = rf.predict(input_data)[0]
st.success(f"Predicted Demand: **{prediction:.0f} units**")

with st.expander("View Raw Data"):
    st.dataframe(filtered.head(500), use_container_width=True)
