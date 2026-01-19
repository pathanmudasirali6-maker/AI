import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="BMW Market Analysis - Professional Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Look
st.markdown("""
    <style>
    :root {
        --primary-color: #0066cc;
        --secondary-color: #003d99;
        --accent-color: #ff6b35;
    }
    
    .main {
        padding: 20px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 12px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        margin-top: 10px;
    }
    
    .metric-label {
        font-size: 14px;
        opacity: 0.9;
        font-weight: 500;
    }
    
    .header-section {
        background: linear-gradient(90deg, #0066cc 0%, #003d99 100%);
        padding: 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .insight-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-left: 4px solid #0066cc;
        border-radius: 4px;
        margin: 10px 0;
    }
    
    .insight-box-warning {
        background-color: #fff3cd;
        padding: 15px;
        border-left: 4px solid #ff6b35;
        border-radius: 4px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load and Validate Data
@st.cache_data
def load_data():
    df = pd.read_csv('bmw.csv')
    df.columns = df.columns.str.strip()
    df = df.dropna()  # Remove null values
    return df

df = load_data()

# Professional Header
st.markdown("""
    <div class="header-section">
        <h1>🚗 BMW Market Intelligence Dashboard</h1>
        <p style="font-size: 16px; margin: 10px 0 0 0;">Comprehensive Data Analysis & Market Insights</p>
        <p style="font-size: 12px; opacity: 0.8; margin: 5px 0 0 0;">Dataset: %d vehicles | Period: %d-%d</p>
    </div>
    """ % (len(df), int(df['year'].min()), int(df['year'].max())), unsafe_allow_html=True)

# Sidebar with Advanced Filters
st.sidebar.header("⚙️ Filter & Analysis Controls")
st.sidebar.markdown("---")

# Model filter
selected_models = st.sidebar.multiselect(
    "🏷️ Select BMW Models:",
    options=sorted(df['model'].unique()),
    default=sorted(df['model'].unique())[:5]
)

# Year filter
year_range = st.sidebar.slider(
    "📅 Select Year Range:",
    min_value=int(df['year'].min()),
    max_value=int(df['year'].max()),
    value=(int(df['year'].min()), int(df['year'].max()))
)

# Price range filter
price_range = st.sidebar.slider(
    "💷 Select Price Range (£):",
    min_value=int(df['price'].min()),
    max_value=int(df['price'].max()),
    value=(int(df['price'].min()), int(df['price'].max())),
    step=1000
)

# Transmission filter
selected_transmission = st.sidebar.multiselect(
    "⚙️ Select Transmission Type:",
    options=df['transmission'].unique(),
    default=df['transmission'].unique()
)

# Fuel type filter
selected_fuel = st.sidebar.multiselect(
    "⛽ Select Fuel Type:",
    options=df['fuelType'].unique(),
    default=df['fuelType'].unique()
)

# Mileage filter
mileage_range = st.sidebar.slider(
    "🛣️ Select Mileage Range (miles):",
    min_value=int(df['mileage'].min()),
    max_value=int(df['mileage'].max()),
    value=(int(df['mileage'].min()), int(df['mileage'].max())),
    step=5000
)

st.sidebar.markdown("---")

# Apply filters with new conditions
filtered_df = df[
    (df['model'].isin(selected_models)) &
    (df['year'] >= year_range[0]) &
    (df['year'] <= year_range[1]) &
    (df['price'] >= price_range[0]) &
    (df['price'] <= price_range[1]) &
    (df['transmission'].isin(selected_transmission)) &
    (df['fuelType'].isin(selected_fuel)) &
    (df['mileage'] >= mileage_range[0]) &
    (df['mileage'] <= mileage_range[1])
]

# Calculate professional metrics
def calculate_metrics(data):
    return {
        'count': len(data),
        'avg_price': data['price'].mean(),
        'median_price': data['price'].median(),
        'price_std': data['price'].std(),
        'avg_mileage': data['mileage'].mean(),
        'avg_mpg': data['mpg'].mean(),
        'avg_engine': data['engineSize'].mean(),
        'avg_tax': data['tax'].mean(),
        'price_min': data['price'].min(),
        'price_max': data['price'].max(),
        'price_range': data['price'].max() - data['price'].min(),
    }

metrics = calculate_metrics(filtered_df)

st.markdown("---")

# Executive Summary & Key Metrics
st.header("📊 Executive Summary - Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("📈 Total Vehicles", f"{metrics['count']:,}", 
              delta=f"{(metrics['count']/len(df)*100):.1f}% of dataset")

with col2:
    st.metric("💷 Avg Price", f"£{metrics['avg_price']:,.0f}", 
              delta=f"Range: £{metrics['price_min']:,} - £{metrics['price_max']:,}")

with col3:
    st.metric("🛣️ Avg Mileage", f"{metrics['avg_mileage']:,.0f} mi", 
              delta=f"Median: {filtered_df['mileage'].median():,.0f}")

with col4:
    st.metric("⛽ Avg MPG", f"{metrics['avg_mpg']:.1f}", 
              delta=f"Engine: {metrics['avg_engine']:.2f}L")

with col5:
    st.metric("💰 Avg Tax", f"£{metrics['avg_tax']:,.0f}", 
              delta=f"Std Dev: £{metrics['price_std']:,.0f}")

st.markdown("---")

# Professional Insights Section
st.header("💡 Market Insights & Analysis")

col1, col2 = st.columns(2)

with col1:
    # Price Analysis Insight
    price_trend = filtered_df.groupby('year')['price'].mean()
    recent_years = price_trend.tail(3)
    if len(recent_years) > 1:
        price_change = ((recent_years.iloc[-1] - recent_years.iloc[0]) / recent_years.iloc[0]) * 100
        direction = "📈 Rising" if price_change > 0 else "📉 Declining"
        st.markdown(f"""
        <div class="insight-box">
        <b>💷 Price Trend Analysis</b><br>
        {direction} by {abs(price_change):.1f}% over recent years<br>
        Current Average: £{metrics['avg_price']:,.0f}<br>
        Market Range: £{metrics['price_range']:,}
        </div>
        """, unsafe_allow_html=True)

with col2:
    # Fuel Type Preference
    fuel_dist = filtered_df['fuelType'].value_counts()
    top_fuel = fuel_dist.index[0]
    top_fuel_pct = (fuel_dist.iloc[0] / len(filtered_df)) * 100
    st.markdown(f"""
    <div class="insight-box">
    <b>⛽ Fuel Type Distribution</b><br>
    Most Common: {top_fuel} ({top_fuel_pct:.1f}%)<br>
    Total Fuel Types: {len(fuel_dist)}<br>
    Total Models: {filtered_df['model'].nunique()}
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Charts Section
st.header("📈 Advanced Analytics & Visualizations")

# Row 1 - Professional Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Price Distribution Analysis by Model (Top 12)")
    price_by_model = filtered_df.groupby('model').agg({
        'price': ['mean', 'count', 'median']
    }).sort_values(('price', 'mean'), ascending=False).head(12)
    price_by_model.columns = ['_'.join(col).strip() for col in price_by_model.columns.values]
    
    fig = px.bar(
        x=price_by_model['price_mean'],
        y=price_by_model.index,
        orientation='h',
        title='Average Price by Model',
        labels={'price_mean': 'Average Price (£)'},
        color=price_by_model['price_mean'],
        color_continuous_scale='Blues',
        text=[f"£{x:,.0f}" for x in price_by_model['price_mean']]
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(height=450, showlegend=False, hovermode='closest')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Mileage vs Price Correlation Analysis")
    correlation = filtered_df['mileage'].corr(filtered_df['price'])
    
    sample_data = filtered_df.sample(min(800, len(filtered_df)))
    fig = px.scatter(
        sample_data,
        x='mileage',
        y='price',
        color='engineSize',
        size='mpg',
        hover_data=['model', 'year', 'transmission', 'fuelType'],
        title=f'Mileage vs Price (Correlation: {correlation:.3f})',
        labels={'mileage': 'Mileage (miles)', 'price': 'Price (£)', 'engineSize': 'Engine Size'},
        color_continuous_scale='Viridis',
        trendline='ols'
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

# Row 2 - Fuel and Transmission
col1, col2 = st.columns(2)

with col1:
    st.subheader("Vehicle Count by Fuel Type Distribution")
    fuel_count = filtered_df['fuelType'].value_counts()
    colors = px.colors.qualitative.Set2
    fig = px.pie(
        values=fuel_count.values,
        names=fuel_count.index,
        title='Fuel Type Market Share',
        color_discrete_sequence=colors,
        hole=0.3
    )
    fig.update_traces(textinfo='label+percent+value')
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Transmission Type Market Analysis")
    trans_analysis = filtered_df.groupby('transmission').agg({
        'price': 'mean',
        'model': 'count',
        'mpg': 'mean'
    }).sort_values('price', ascending=False)
    
    fig = go.Figure(data=[
        go.Bar(
            x=trans_analysis.index,
            y=trans_analysis['price'],
            marker_color='#636EFA',
            name='Avg Price',
            text=[f"£{x:,.0f}" for x in trans_analysis['price']],
            textposition='outside'
        )
    ])
    fig.update_layout(
        title='Average Price by Transmission Type',
        xaxis_title='Transmission Type',
        yaxis_title='Average Price (£)',
        height=450,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

# Row 3 - Year Analysis
col1, col2 = st.columns(2)

with col1:
    st.subheader("Historical Price Trend Analysis")
    price_by_year = filtered_df.groupby('year').agg({
        'price': ['mean', 'median', 'count']
    }).sort_index()
    price_by_year.columns = ['_'.join(col).strip() for col in price_by_year.columns.values]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_by_year.index,
        y=price_by_year['price_mean'],
        mode='lines+markers',
        name='Average Price',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=price_by_year.index,
        y=price_by_year['price_median'],
        mode='lines+markers',
        name='Median Price',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    fig.update_layout(
        title='Price Trends Over Years',
        xaxis_title='Year',
        yaxis_title='Price (£)',
        height=450,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Vehicle Count by Year")
    vehicle_count = filtered_df.groupby('year').size()
    
    fig = px.bar(
        x=vehicle_count.index,
        y=vehicle_count.values,
        title='Vehicle Availability by Year',
        labels={'x': 'Year', 'y': 'Count'},
        color=vehicle_count.values,
        color_continuous_scale='Greens',
        text=vehicle_count.values
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Row 4 - Performance Metrics
col1, col2 = st.columns(2)

with col1:
    st.subheader("MPG Distribution Analysis")
    fig = px.histogram(
        filtered_df,
        x='mpg',
        nbins=40,
        title='Fuel Efficiency Distribution',
        labels={'mpg': 'Miles Per Gallon (MPG)', 'count': 'Frequency'},
        color_discrete_sequence=['#636EFA']
    )
    fig.add_vline(x=filtered_df['mpg'].mean(), line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {filtered_df['mpg'].mean():.1f}")
    fig.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Engine Size vs Price Analysis")
    engine_analysis = filtered_df.groupby(pd.cut(filtered_df['engineSize'], bins=8)).agg({
        'price': ['mean', 'count'],
        'mpg': 'mean'
    }).round(2)
    
    fig = px.scatter(
        filtered_df.sample(min(500, len(filtered_df))),
        x='engineSize',
        y='price',
        size='mpg',
        color='mileage',
        hover_data=['model', 'year', 'fuelType'],
        title='Engine Size vs Price Relationship',
        labels={'engineSize': 'Engine Size (L)', 'price': 'Price (£)', 'mileage': 'Mileage'},
        color_continuous_scale='RdYlBu_r',
        trendline='ols'
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Statistical Analysis Section
st.header("📊 Statistical Analysis & Correlation")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Price Statistics")
    st.write(f"""
    **Minimum:** £{filtered_df['price'].min():,.0f}
    
    **Maximum:** £{filtered_df['price'].max():,.0f}
    
    **Median:** £{filtered_df['price'].median():,.0f}
    
    **Mean:** £{filtered_df['price'].mean():,.0f}
    
    **Std Dev:** £{filtered_df['price'].std():,.0f}
    
    **Skewness:** {skew(filtered_df['price']):.3f}
    
    **Kurtosis:** {kurtosis(filtered_df['price']):.3f}
    """)

with col2:
    st.subheader("Mileage Statistics")
    st.write(f"""
    **Minimum:** {filtered_df['mileage'].min():,.0f} mi
    
    **Maximum:** {filtered_df['mileage'].max():,.0f} mi
    
    **Median:** {filtered_df['mileage'].median():,.0f} mi
    
    **Mean:** {filtered_df['mileage'].mean():,.0f} mi
    
    **Std Dev:** {filtered_df['mileage'].std():,.0f} mi
    
    **Skewness:** {skew(filtered_df['mileage']):.3f}
    
    **Kurtosis:** {kurtosis(filtered_df['mileage']):.3f}
    """)

with col3:
    st.subheader("Performance Metrics")
    st.write(f"""
    **Avg MPG:** {filtered_df['mpg'].mean():.2f}
    
    **MPG Range:** {filtered_df['mpg'].min():.1f} - {filtered_df['mpg'].max():.1f}
    
    **Avg Engine:** {filtered_df['engineSize'].mean():.2f}L
    
    **Avg Tax:** £{filtered_df['tax'].mean():,.0f}
    
    **Total Models:** {filtered_df['model'].nunique()}
    
    **Fuel Types:** {filtered_df['fuelType'].nunique()}
    
    **Years Covered:** {int(filtered_df['year'].max() - filtered_df['year'].min())} years
    """)

st.markdown("---")

# Correlation Matrix
st.subheader("🔗 Correlation Matrix Analysis")
correlation_cols = ['price', 'mileage', 'mpg', 'tax', 'engineSize', 'year']
correlation_matrix = filtered_df[correlation_cols].corr()

fig = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=correlation_cols,
    y=correlation_cols,
    colorscale='RdBu',
    zmid=0,
    text=np.round(correlation_matrix.values, 2),
    texttemplate='%{text:.2f}',
    textfont={"size": 10},
    colorbar=dict(title="Correlation")
))
fig.update_layout(height=500, title_text="Correlation Heatmap - Numerical Variables")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Premium Models Analysis
st.header("🏆 Top Models & Benchmarking Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Top 10 Most Expensive Models")
    top_expensive = filtered_df.groupby('model').agg({
        'price': 'mean',
        'model': 'count'
    }).sort_values('price', ascending=False).head(10)
    top_expensive.columns = ['Avg Price', 'Count']
    top_expensive['Avg Price'] = top_expensive['Avg Price'].apply(lambda x: f"£{x:,.0f}")
    st.dataframe(top_expensive, use_container_width=True)

with col2:
    st.subheader("Most Fuel Efficient Models")
    top_mpg = filtered_df.groupby('model').agg({
        'mpg': 'mean',
        'model': 'count'
    }).sort_values('mpg', ascending=False).head(10)
    top_mpg.columns = ['Avg MPG', 'Count']
    top_mpg['Avg MPG'] = top_mpg['Avg MPG'].apply(lambda x: f"{x:.1f}")
    st.dataframe(top_mpg, use_container_width=True)

with col3:
    st.subheader("Lowest Mileage Models")
    low_mileage = filtered_df.groupby('model').agg({
        'mileage': 'mean',
        'model': 'count'
    }).sort_values('mileage', ascending=True).head(10)
    low_mileage.columns = ['Avg Mileage', 'Count']
    low_mileage['Avg Mileage'] = low_mileage['Avg Mileage'].apply(lambda x: f"{x:,.0f} mi")
    st.dataframe(low_mileage, use_container_width=True)

st.markdown("---")

# Market Segmentation
st.header("📍 Market Segmentation Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Transmission Type Market Share & Performance")
    trans_segment = filtered_df.groupby('transmission').agg({
        'model': 'count',
        'price': 'mean',
        'mpg': 'mean',
        'mileage': 'mean'
    }).round(2)
    trans_segment.columns = ['Count', 'Avg Price', 'Avg MPG', 'Avg Mileage']
    trans_segment['% Share'] = (trans_segment['Count'] / trans_segment['Count'].sum() * 100).round(1)
    st.dataframe(trans_segment, use_container_width=True)

with col2:
    st.subheader("Fuel Type Market Analysis")
    fuel_segment = filtered_df.groupby('fuelType').agg({
        'model': 'count',
        'price': 'mean',
        'mpg': 'mean',
        'tax': 'mean'
    }).round(2)
    fuel_segment.columns = ['Count', 'Avg Price', 'Avg MPG', 'Avg Tax']
    fuel_segment['% Share'] = (fuel_segment['Count'] / fuel_segment['Count'].sum() * 100).round(1)
    st.dataframe(fuel_segment, use_container_width=True)

st.markdown("---")

# Year-over-Year Analysis
st.header("📅 Year-over-Year Performance")

year_analysis = filtered_df.groupby('year').agg({
    'model': 'count',
    'price': ['mean', 'median'],
    'mileage': 'mean',
    'mpg': 'mean',
    'tax': 'mean'
}).round(2)
year_analysis.columns = ['Vehicle Count', 'Avg Price', 'Median Price', 'Avg Mileage', 'Avg MPG', 'Avg Tax']
st.dataframe(year_analysis, use_container_width=True)

st.markdown("---")

# Data Table with Export Options
st.header("📑 Detailed Data Table & Export")

# Search and filter the table
search_term = st.text_input("🔍 Search in data (model, fuel type, transmission):")
if search_term:
    table_df = filtered_df[
        filtered_df['model'].str.contains(search_term, case=False) |
        filtered_df['fuelType'].str.contains(search_term, case=False) |
        filtered_df['transmission'].str.contains(search_term, case=False)
    ]
else:
    table_df = filtered_df

st.write(f"Showing {len(table_df)} records")
st.dataframe(table_df.sort_values('price', ascending=False), use_container_width=True, height=500)

# Summary Statistics
st.markdown("---")
st.header("📊 Comprehensive Summary Statistics")

summary_stats = filtered_df.describe().round(2)
st.dataframe(summary_stats, use_container_width=True)

st.markdown("---")

# Professional Recommendations
st.header("💼 Professional Insights & Recommendations")

col1, col2 = st.columns(2)

with col1:
    # Value Analysis
    price_per_mpg = filtered_df['price'] / filtered_df['mpg']
    best_value_idx = price_per_mpg.idxmin()
    best_value = filtered_df.loc[best_value_idx]
    
    st.markdown(f"""
    <div class="insight-box">
    <b>🎯 Best Value Proposition</b><br>
    Model: {best_value['model']}<br>
    Price: £{best_value['price']:,.0f}<br>
    MPG: {best_value['mpg']:.1f} (£{best_value['price']/best_value['mpg']:.2f} per MPG)<br>
    Year: {int(best_value['year'])}
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Budget Analysis
    budget_vehicles = filtered_df[filtered_df['price'] < filtered_df['price'].quantile(0.25)]
    if len(budget_vehicles) > 0:
        avg_budget_mpg = budget_vehicles['mpg'].mean()
        avg_budget_mileage = budget_vehicles['mileage'].mean()
        
        st.markdown(f"""
        <div class="insight-box">
        <b>💰 Budget Segment Analysis (Bottom 25%)</b><br>
        Average Price: £{budget_vehicles['price'].mean():,.0f}<br>
        Average MPG: {avg_budget_mpg:.1f}<br>
        Average Mileage: {avg_budget_mileage:,.0f} miles<br>
        Vehicle Count: {len(budget_vehicles)}
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Data Quality Report
st.header("✅ Data Quality & Completeness Report")

col1, col2, col3, col4 = st.columns(4)

with col1:
    null_count = filtered_df.isnull().sum().sum()
    st.metric("Missing Values", null_count, delta="0%" if null_count == 0 else "⚠️")

with col2:
    duplicate_count = filtered_df.duplicated().sum()
    st.metric("Duplicate Records", duplicate_count, delta="0%" if duplicate_count == 0 else "⚠️")

with col3:
    data_completeness = ((len(filtered_df) * len(filtered_df.columns) - null_count) / 
                         (len(filtered_df) * len(filtered_df.columns))) * 100
    st.metric("Data Completeness", f"{data_completeness:.1f}%", delta="✅ Excellent")

with col4:
    outlier_estimate = len(filtered_df[(filtered_df['price'] < 
                                        filtered_df['price'].quantile(0.01)) |
                                       (filtered_df['price'] > 
                                        filtered_df['price'].quantile(0.99))])
    st.metric("Potential Outliers", outlier_estimate, delta=f"{(outlier_estimate/len(filtered_df)*100):.1f}%")

st.markdown("---")

# Footer
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px; font-size: 12px;'>
    <b>BMW Market Intelligence Dashboard</b><br>
    Professional Data Analysis & Market Insights<br>
    Generated: January 2026 | Data Points Analyzed: %d vehicles | Report Version: 2.0
    <br><br>
    <i>This dashboard provides comprehensive market analysis for BMW vehicles including pricing trends, 
    fuel efficiency analysis, transmission preferences, and market segmentation insights.</i>
</div>
""" % len(df), unsafe_allow_html=True)
