import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Pakistan Crime Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('248a92b9-5817-43f2-b0e7-6d45a45a8ee9.csv')

df = load_data()

# ============================================================================
# ADVANCED SIDEBAR CONFIGURATION
# ============================================================================
st.sidebar.title("🎛️ Advanced Dashboard Controls")
st.sidebar.markdown("---")

# Create expandable sections in sidebar
with st.sidebar.expander("📅 Time Period Settings", expanded=True):
    selected_years = st.slider(
        "Select Year Range:",
        int(df['Year'].min()),
        int(df['Year'].max()),
        (int(df['Year'].min()), int(df['Year'].max())),
        key='year_slider'
    )
    
    # Quick date presets
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔲 All Years", use_container_width=True):
            selected_years = (int(df['Year'].min()), int(df['Year'].max()))
    with col2:
        if st.button("📊 Last 3 Yrs", use_container_width=True):
            selected_years = (2015, 2017)
    with col3:
        if st.button("📌 2017 Only", use_container_width=True):
            selected_years = (2017, 2017)

provinces_list = ['Punjab', 'Sindh', 'KP', 'Balochistan', 'Islamabad', 'Railways', 'G.B', 'AJK']

with st.sidebar.expander("🗺️ Provincial Filters", expanded=True):
    st.write("**Select Provinces:**")
    selected_provinces = st.multiselect(
        "Provinces to analyze:",
        provinces_list,
        default=['Punjab', 'Sindh', 'KP'],
        key='prov_filter'
    )
    
    # Quick selection buttons
    st.write("**Quick Select:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("All Provinces", use_container_width=True, key='all_prov'):
            selected_provinces = provinces_list
    with col2:
        if st.button("Major Cities", use_container_width=True, key='major_prov'):
            selected_provinces = ['Punjab', 'Sindh', 'KP']
    with col3:
        if st.button("Clear All", use_container_width=True, key='clear_prov'):
            selected_provinces = []

crime_categories = df[df['Offence'] != 'TOTAL RECORDED CRIME']['Offence'].unique().tolist()

with st.sidebar.expander("🚨 Crime Type Filters", expanded=True):
    st.write("**Select Crime Types:**")
    selected_crimes = st.multiselect(
        "Crime categories to analyze:",
        crime_categories,
        default=crime_categories[:5],
        key='crime_filter'
    )
    
    # Quick selection
    st.write("**Quick Categories:**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Violent Crimes", use_container_width=True, key='violent'):
            selected_crimes = ['Murder', 'Attempt to Murder', 'Kidnapping /Abduction', 'Robbery']
    with col2:
        if st.button("Property Crimes", use_container_width=True, key='property'):
            selected_crimes = ['Dacoity', 'Burglary', 'Cattle Theft', 'Other Theft']

# Advanced Analytics Section
with st.sidebar.expander("📊 Analytics Options", expanded=True):
    st.write("**Visualization Settings:**")
    
    chart_theme = st.selectbox(
        "Chart Theme:",
        ["Plotly", "Dark", "Light"],
        key='theme_select'
    )
    
    show_trend_line = st.checkbox("📈 Show Trend Lines", value=True, key='trend_line')
    show_statistics = st.checkbox("📋 Show Statistics", value=True, key='show_stats')
    show_comparison = st.checkbox("🔄 Show Year Comparisons", value=True, key='comparison')
    smooth_data = st.checkbox("✨ Smooth Data", value=False, key='smooth')

# Insights Section
with st.sidebar.expander("💡 Quick Insights", expanded=False):
    df_filtered_temp = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]
    
    total_crimes_insight = df_filtered_temp[df_filtered_temp['Offence'] == 'TOTAL RECORDED CRIME']['Pakistan'].sum()
    
    # Top crime
    top_crime = df_filtered_temp[df_filtered_temp['Offence'] != 'TOTAL RECORDED CRIME'].groupby('Offence')['Pakistan'].sum().idxmax()
    top_crime_count = df_filtered_temp[df_filtered_temp['Offence'] != 'TOTAL RECORDED CRIME'].groupby('Offence')['Pakistan'].sum().max()
    
    # Most affected province
    most_affected = df_filtered_temp[df_filtered_temp['Offence'] == 'TOTAL RECORDED CRIME'][provinces_list].sum().idxmax()
    
    st.metric("📊 Total Crimes", f"{int(total_crimes_insight):,}")
    st.metric("🚨 Top Crime", f"{top_crime} ({int(top_crime_count):,})")
    st.metric("🗺️ Most Affected", most_affected)
    
    st.markdown("---")
    
    # Year comparison
    if len(selected_years) > 0:
        crime_first_year = df_filtered_temp[
            (df_filtered_temp['Year'] == selected_years[0]) & 
            (df_filtered_temp['Offence'] == 'TOTAL RECORDED CRIME')
        ]['Pakistan'].sum()
        
        crime_last_year = df_filtered_temp[
            (df_filtered_temp['Year'] == selected_years[1]) & 
            (df_filtered_temp['Offence'] == 'TOTAL RECORDED CRIME')
        ]['Pakistan'].sum()
        
        if crime_first_year > 0:
            change_pct = ((crime_last_year - crime_first_year) / crime_first_year) * 100
            st.metric(
                f"Change ({selected_years[0]}-{selected_years[1]})",
                f"{change_pct:+.1f}%",
                delta_color="inverse"
            )

# Data Export Section
with st.sidebar.expander("💾 Export Data", expanded=False):
    st.write("**Download Options:**")
    
    df_filtered_temp = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]
    
    # Export filtered data
    csv_data = df_filtered_temp.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv_data,
        file_name=f"crime_data_{selected_years[0]}-{selected_years[1]}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # Export summary statistics
    summary_stats = df_filtered_temp[df_filtered_temp['Offence'] == 'TOTAL RECORDED CRIME'].describe()
    stats_csv = summary_stats.to_csv()
    st.download_button(
        label="📊 Download Statistics (CSV)",
        data=stats_csv,
        file_name="crime_statistics.csv",
        mime="text/csv",
        use_container_width=True
    )

# Help Section
with st.sidebar.expander("❓ Help & Information", expanded=False):
    st.markdown("""
    ### 📖 Dashboard Guide
    
    **Time Period:** Select the years to analyze
    - Use quick buttons for preset ranges
    
    **Provinces:** Choose which provinces to include
    - Major Cities preset includes Punjab, Sindh, KP
    
    **Crime Types:** Select specific crime categories
    - Violent Crimes & Property Crimes quick filters
    
    **Analytics:** Customize chart appearance
    - Change themes and enable trend lines
    
    **Insights:** See key statistics at a glance
    - Quick metrics for selected data
    
    **Export:** Download data for external analysis
    """)
    
    st.markdown("---")
    st.markdown("**Data Period:** 2012-2017")
    st.markdown("**Source:** Pakistan Crime Statistics")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='text-align: center; font-size: 11px; color: #888;'>"
    "🚀 Advanced Dashboard v2.0<br>"
    "Built with Streamlit & Plotly"
    "</div>",
    unsafe_allow_html=True
)

# Filter data
df_filtered = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]

# ============================================================================
# MAIN HEADER
# ============================================================================
st.markdown("""
    <h1 style='text-align: center; color: #1f77b4;'>
    📊 PAKISTAN CRIME ANALYSIS DASHBOARD
    </h1>
    <h3 style='text-align: center; color: #666;'>
    Comprehensive Crime Statistics & Trends (2012-2017)
    </h3>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# KEY METRICS
# ============================================================================
st.subheader("📈 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

total_crimes = df_filtered[df_filtered['Offence'] == 'TOTAL RECORDED CRIME']['Pakistan'].sum()
avg_yearly = df_filtered[df_filtered['Offence'] == 'TOTAL RECORDED CRIME']['Pakistan'].mean()
crime_categories_count = df_filtered[df_filtered['Offence'] != 'TOTAL RECORDED CRIME']['Offence'].nunique()
years_count = df_filtered['Year'].nunique()

with col1:
    st.metric(
        "Total Crimes",
        f"{int(total_crimes):,}",
        delta=f"{years_count} years",
        delta_color="off"
    )

with col2:
    st.metric(
        "Avg Annual Crimes",
        f"{int(avg_yearly):,}",
        delta="Per Year",
        delta_color="off"
    )

with col3:
    st.metric(
        "Crime Categories",
        crime_categories_count,
        delta="Types",
        delta_color="off"
    )

with col4:
    st.metric(
        "Data Period",
        f"{years_count} Years",
        delta=f"{df_filtered['Year'].min()}-{df_filtered['Year'].max()}",
        delta_color="off"
    )

st.markdown("---")

# ============================================================================
# TAB NAVIGATION
# ============================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Overview", 
    "📍 Provincial Analysis", 
    "🚨 Crime Types", 
    "📉 Trends & Predictions", 
    "🔬 Deep Analysis",
    "📈 Statistical Insights",
    "🎯 Crime Patterns",
    "📋 Data Explorer"
])

# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================
with tab1:
    st.subheader("National Crime Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Total crime trend
        yearly_trend = df_filtered[df_filtered['Offence'] == 'TOTAL RECORDED CRIME'][['Year', 'Pakistan']].sort_values('Year')
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=yearly_trend['Year'],
            y=yearly_trend['Pakistan'],
            mode='lines+markers',
            name='Total Crimes',
            line=dict(color='#d62728', width=3),
            marker=dict(size=10),
            fill='tozeroy'
        ))
        fig_trend.update_layout(
            title="Crime Trend Over Years",
            xaxis_title="Year",
            yaxis_title="Number of Crimes",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        # Top 8 crime categories
        crime_dist = df_filtered[df_filtered['Offence'] != 'TOTAL RECORDED CRIME'].groupby('Offence')['Pakistan'].sum().sort_values(ascending=False).head(8)
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=crime_dist.index,
            values=crime_dist.values,
            hovertemplate='<b>%{label}</b><br>Cases: %{value:,.0f}<extra></extra>'
        )])
        fig_pie.update_layout(
            title="Crime Distribution (Top 8)",
            height=400
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Crime by category bar chart
    st.subheader("All Crime Categories Total")
    crime_summary = df_filtered[df_filtered['Offence'] != 'TOTAL RECORDED CRIME'].groupby('Offence')['Pakistan'].sum().sort_values(ascending=False)
    
    fig_bar = go.Figure(data=[go.Bar(
        y=crime_summary.index,
        x=crime_summary.values,
        orientation='h',
        marker_color=crime_summary.values,
        marker_colorscale='Reds'
    )])
    fig_bar.update_layout(
        title="Total Crimes by Type (All Years)",
        xaxis_title="Number of Cases",
        height=500,
        hovermode='y unified'
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ============================================================================
# TAB 2: PROVINCIAL ANALYSIS
# ============================================================================
with tab2:
    st.subheader("Provincial Crime Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Provincial totals
        prov_totals = df_filtered[df_filtered['Offence'] == 'TOTAL RECORDED CRIME'][provinces_list].sum().sort_values(ascending=False)
        
        fig_prov = go.Figure(data=[go.Bar(
            x=prov_totals.index,
            y=prov_totals.values,
            marker_color=prov_totals.values,
            marker_colorscale='Blues'
        )])
        fig_prov.update_layout(
            title="Total Crimes by Province",
            yaxis_title="Number of Crimes",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig_prov, use_container_width=True)
    
    with col2:
        # Year-wise provincial trends
        fig_trends = go.Figure()
        for province in selected_provinces:
            prov_yearly = df_filtered[df_filtered['Offence'] == 'TOTAL RECORDED CRIME'][['Year', province]].sort_values('Year')
            fig_trends.add_trace(go.Scatter(
                x=prov_yearly['Year'],
                y=prov_yearly[province],
                mode='lines+markers',
                name=province,
                marker=dict(size=6)
            ))
        fig_trends.update_layout(
            title="Crime Trends by Selected Provinces",
            xaxis_title="Year",
            yaxis_title="Number of Crimes",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_trends, use_container_width=True)
    
    # Provincial comparison heatmap
    st.subheader("Provincial Crime Heatmap Over Years")
    prov_crime_matrix = df_filtered[df_filtered['Offence'] == 'TOTAL RECORDED CRIME'].set_index('Year')[provinces_list]
    
    fig_heat = go.Figure(data=go.Heatmap(
        z=prov_crime_matrix.values,
        x=prov_crime_matrix.columns,
        y=prov_crime_matrix.index,
        colorscale='RdYlGn_r'
    ))
    fig_heat.update_layout(
        title="Provincial Crimes Heatmap",
        height=400
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ============================================================================
# TAB 3: CRIME TYPES
# ============================================================================
with tab3:
    st.subheader("Detailed Crime Type Analysis")
    
    # Top crimes over years
    st.subheader("Top Crime Types Over Years")
    top_crimes = df_filtered[df_filtered['Offence'] != 'TOTAL RECORDED CRIME'].groupby('Offence')['Pakistan'].sum().sort_values(ascending=False).head(6).index
    crime_yearly = df_filtered[df_filtered['Offence'].isin(top_crimes)].pivot_table(
        values='Pakistan',
        index='Year',
        columns='Offence',
        aggfunc='sum'
    )
    
    fig_crime = go.Figure()
    for crime in crime_yearly.columns:
        fig_crime.add_trace(go.Scatter(
            x=crime_yearly.index,
            y=crime_yearly[crime],
            mode='lines+markers',
            name=crime
        ))
    fig_crime.update_layout(
        title="Top 6 Crime Types Over Time",
        xaxis_title="Year",
        yaxis_title="Number of Cases",
        hovermode='x unified',
        height=450
    )
    st.plotly_chart(fig_crime, use_container_width=True)
    
    # Specific crime analysis
    st.subheader("Deep Dive: Crime Type Comparison")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Murder analysis
        murder_data = df_filtered[df_filtered['Offence'] == 'Murder'][['Year', 'Pakistan']].sort_values('Year')
        fig_murder = go.Figure()
        fig_murder.add_trace(go.Bar(x=murder_data['Year'], y=murder_data['Pakistan'], name='Murder', marker_color='darkred'))
        fig_murder.update_layout(title="Murder Cases", height=350, showlegend=False)
        st.plotly_chart(fig_murder, use_container_width=True)
    
    with col2:
        # Kidnapping analysis
        kidnap_data = df_filtered[df_filtered['Offence'] == 'Kidnapping /Abduction'][['Year', 'Pakistan']].sort_values('Year')
        fig_kidnap = go.Figure()
        fig_kidnap.add_trace(go.Bar(x=kidnap_data['Year'], y=kidnap_data['Pakistan'], name='Kidnapping', marker_color='orange'))
        fig_kidnap.update_layout(title="Kidnapping/Abduction", height=350, showlegend=False)
        st.plotly_chart(fig_kidnap, use_container_width=True)
    
    with col3:
        # Robbery analysis
        robbery_data = df_filtered[df_filtered['Offence'] == 'Robbery'][['Year', 'Pakistan']].sort_values('Year')
        fig_robbery = go.Figure()
        fig_robbery.add_trace(go.Bar(x=robbery_data['Year'], y=robbery_data['Pakistan'], name='Robbery', marker_color='purple'))
        fig_robbery.update_layout(title="Robbery Cases", height=350, showlegend=False)
        st.plotly_chart(fig_robbery, use_container_width=True)

# ============================================================================
# TAB 4: TRENDS & PREDICTIONS
# ============================================================================
with tab4:
    st.subheader("Crime Trend Analysis")
    
    # Year-over-year change
    st.subheader("Year-over-Year Change Analysis")
    yearly_totals = df_filtered[df_filtered['Offence'] == 'TOTAL RECORDED CRIME'].set_index('Year')['Pakistan'].sort_index()
    yoy_change = yearly_totals.pct_change() * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_yoy = go.Figure()
        fig_yoy.add_trace(go.Bar(
            x=yoy_change.index,
            y=yoy_change.values,
            marker_color=['red' if x < 0 else 'green' for x in yoy_change.values]
        ))
        fig_yoy.update_layout(
            title="Year-over-Year Crime Change (%)",
            xaxis_title="Year",
            yaxis_title="Change (%)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_yoy, use_container_width=True)
    
    with col2:
        # Statistical summary
        st.write("### Crime Statistics Summary")
        stats_data = {
            'Metric': [
                'Total Crimes (Period)',
                'Average Annual Crimes',
                'Median Annual Crimes',
                'Highest Year',
                'Lowest Year',
                'Std Deviation'
            ],
            'Value': [
                f"{int(yearly_totals.sum()):,}",
                f"{int(yearly_totals.mean()):,}",
                f"{int(yearly_totals.median()):,}",
                f"{int(yearly_totals.max()):,} ({yearly_totals.idxmax()})",
                f"{int(yearly_totals.min()):,} ({yearly_totals.idxmin()})",
                f"{int(yearly_totals.std()):,}"
            ]
        }
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
    
    # Crime growth comparison
    st.subheader("Crime Growth by Category")
    crime_growth = []
    for crime in crime_categories[:10]:
        crime_2012 = df[(df['Year'] == df['Year'].min()) & (df['Offence'] == crime)]['Pakistan'].values
        crime_2017 = df[(df['Year'] == df['Year'].max()) & (df['Offence'] == crime)]['Pakistan'].values
        
        if len(crime_2012) > 0 and len(crime_2017) > 0:
            growth = ((crime_2017[0] - crime_2012[0]) / crime_2012[0]) * 100
            crime_growth.append({'Crime': crime, 'Growth (%)': growth})
    
    growth_df = pd.DataFrame(crime_growth).sort_values('Growth (%)')
    
    fig_growth = go.Figure(data=[go.Bar(
        y=growth_df['Crime'],
        x=growth_df['Growth (%)'],
        orientation='h',
        marker_color=['red' if x < 0 else 'green' for x in growth_df['Growth (%)']]
    )])
    fig_growth.update_layout(
        title="Crime Growth Rate (2012-2017)",
        xaxis_title="Growth (%)",
        height=400
    )
    st.plotly_chart(fig_growth, use_container_width=True)
    
    # Advanced trend analysis with forecasting
    st.subheader("📈 Trend Forecasting & Projections")
    
    yearly_totals_full = df[df['Offence'] == 'TOTAL RECORDED CRIME'].set_index('Year')['Pakistan'].sort_index()
    years_array = yearly_totals_full.index.values
    crimes_array = yearly_totals_full.values
    
    # Polynomial fit for trend
    z = np.polyfit(years_array, crimes_array, 2)
    p = np.poly1d(z)
    
    # Create forecast
    future_years = np.array([2018, 2019, 2020])
    forecast_values = p(future_years)
    
    fig_forecast = go.Figure()
    
    # Historical data
    fig_forecast.add_trace(go.Scatter(
        x=years_array,
        y=crimes_array,
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))
    
    # Advanced forecasting with multiple models
    st.subheader("🔮 Advanced Crime Prediction Models")
    
    yearly_totals_full = df[df['Offence'] == 'TOTAL RECORDED CRIME'].set_index('Year')['Pakistan'].sort_index()
    years_array = yearly_totals_full.index.values
    crimes_array = yearly_totals_full.values
    
    # Calculate forecasts using different models
    future_years = np.array([2018, 2019, 2020])
    
    # Model 1: Polynomial (Degree 2)
    z_poly = np.polyfit(years_array, crimes_array, 2)
    p_poly = np.poly1d(z_poly)
    forecast_poly = p_poly(future_years)
    
    # Model 2: Linear Regression
    z_linear = np.polyfit(years_array, crimes_array, 1)
    p_linear = np.poly1d(z_linear)
    forecast_linear = p_linear(future_years)
    
    # Model 3: Exponential (if data supports it)
    try:
        popt_exp, _ = np.polyfit(years_array, np.log(crimes_array), 1), None
        forecast_exp = np.exp(popt_exp[0] * future_years + popt_exp[1])
    except:
        forecast_exp = forecast_poly  # Fallback
    
    # Calculate confidence intervals using residuals
    residuals_poly = crimes_array - p_poly(years_array)
    std_error = np.std(residuals_poly)
    confidence_interval = 1.96 * std_error  # 95% CI
    
    forecast_upper = forecast_poly + confidence_interval
    forecast_lower = forecast_poly - confidence_interval
    
    # Create advanced forecast visualization
    fig_advanced = go.Figure()
    
    # Historical data
    fig_advanced.add_trace(go.Scatter(
        x=years_array,
        y=crimes_array,
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='blue', width=3),
        marker=dict(size=10)
    ))
    
    # Polynomial forecast with confidence band
    forecast_years_full = np.concatenate([[years_array[-1]], future_years])
    forecast_values_full = np.concatenate([[crimes_array[-1]], forecast_poly])
    upper_full = np.concatenate([[crimes_array[-1]], forecast_upper])
    lower_full = np.concatenate([[crimes_array[-1]], forecast_lower])
    
    fig_advanced.add_trace(go.Scatter(
        x=forecast_years_full,
        y=forecast_values_full,
        mode='lines+markers',
        name='Polynomial Forecast',
        line=dict(color='red', width=3),
        marker=dict(size=8)
    ))
    
    # Confidence interval band
    fig_advanced.add_trace(go.Scatter(
        x=forecast_years_full,
        y=upper_full,
        mode='lines',
        name='95% Upper CI',
        line=dict(color='rgba(255,0,0,0)'),
        showlegend=False
    ))
    
    fig_advanced.add_trace(go.Scatter(
        x=forecast_years_full,
        y=lower_full,
        mode='lines',
        name='95% Confidence Interval',
        line=dict(color='rgba(255,0,0,0)'),
        fillcolor='rgba(255,0,0,0.2)',
        fill='tonexty'
    ))
    
    # Linear forecast
    forecast_lin_full = np.concatenate([[crimes_array[-1]], forecast_linear])
    fig_advanced.add_trace(go.Scatter(
        x=forecast_years_full,
        y=forecast_lin_full,
        mode='lines',
        name='Linear Forecast',
        line=dict(color='orange', width=2, dash='dash')
    ))
    
    fig_advanced.update_layout(
        title="Advanced Crime Forecast with Confidence Intervals",
        xaxis_title="Year",
        yaxis_title="Number of Crimes",
        hovermode='x unified',
        height=450
    )
    st.plotly_chart(fig_advanced, use_container_width=True)
    
    # Forecast comparison table
    st.subheader("📊 Forecast Comparison: Multiple Models")
    
    forecast_comparison = pd.DataFrame({
        'Year': future_years,
        'Polynomial': forecast_poly.astype(int),
        'Linear': forecast_linear.astype(int),
        'Exponential': forecast_exp.astype(int),
        '95% Upper CI': forecast_upper.astype(int),
        '95% Lower CI': forecast_lower.astype(int)
    })
    
    # Add model metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Polynomial model R² value (approximation)
        ss_res = np.sum((crimes_array - p_poly(years_array)) ** 2)
        ss_tot = np.sum((crimes_array - np.mean(crimes_array)) ** 2)
        r2_poly = 1 - (ss_res / ss_tot)
        st.metric("Polynomial R²", f"{r2_poly:.4f}", help="Goodness of fit (0-1)")
    
    with col2:
        ss_res_lin = np.sum((crimes_array - p_linear(years_array)) ** 2)
        r2_linear = 1 - (ss_res_lin / ss_tot)
        st.metric("Linear R²", f"{r2_linear:.4f}", help="Goodness of fit (0-1)")
    
    with col3:
        mae = np.mean(np.abs(residuals_poly))
        st.metric("Mean Abs Error", f"{int(mae):,}", help="Average forecast error")
    
    st.dataframe(forecast_comparison, use_container_width=True, hide_index=True)
    
    # Forecast details and confidence levels
    st.subheader("📈 Forecast Confidence Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    for idx, year in enumerate(future_years):
        with [col1, col2, col3][idx]:
            st.write(f"### {int(year)}")
            st.write(f"**Point Estimate:** {int(forecast_poly[idx]):,}")
            st.write(f"**Range:** {int(forecast_lower[idx]):,} - {int(forecast_upper[idx]):,}")
            change_from_2017 = ((forecast_poly[idx] - crimes_array[-1]) / crimes_array[-1]) * 100
            st.write(f"**Change:** {change_from_2017:+.1f}%")
    
    # Scenario analysis
    st.subheader("🎯 Scenario Analysis")
    
    scenario_col1, scenario_col2 = st.columns(2)
    
    with scenario_col1:
        st.write("### Best Case Scenario")
        st.write("• 5% annual improvement")
        best_case_2020 = crimes_array[-1] * (0.95 ** 3)
        st.write(f"**2020 Estimate:** {int(best_case_2020):,}")
        st.write(f"**Improvement:** {((crimes_array[-1] - best_case_2020) / crimes_array[-1] * 100):+.1f}%")
        
        st.write("### Optimistic Scenario")
        st.write("• 2% annual improvement")
        optimistic_2020 = crimes_array[-1] * (0.98 ** 3)
        st.write(f"**2020 Estimate:** {int(optimistic_2020):,}")
        st.write(f"**Improvement:** {((crimes_array[-1] - optimistic_2020) / crimes_array[-1] * 100):+.1f}%")
    
    with scenario_col2:
        st.write("### Pessimistic Scenario")
        st.write("• 2% annual increase")
        pessimistic_2020 = crimes_array[-1] * (1.02 ** 3)
        st.write(f"**2020 Estimate:** {int(pessimistic_2020):,}")
        st.write(f"**Increase:** {((pessimistic_2020 - crimes_array[-1]) / crimes_array[-1] * 100):+.1f}%")
        
        st.write("### Worst Case Scenario")
        st.write("• 5% annual increase")
        worst_case_2020 = crimes_array[-1] * (1.05 ** 3)
        st.write(f"**2020 Estimate:** {int(worst_case_2020):,}")
        st.write(f"**Increase:** {((worst_case_2020 - crimes_array[-1]) / crimes_array[-1] * 100):+.1f}%")
    
    # Scenario visualization
    scenarios = {
        'Best Case': int(best_case_2020),
        'Optimistic': int(optimistic_2020),
        'Forecast': int(forecast_poly[-1]),
        'Pessimistic': int(pessimistic_2020),
        'Worst Case': int(worst_case_2020)
    }
    
    fig_scenarios = go.Figure(data=[go.Bar(
        x=list(scenarios.keys()),
        y=list(scenarios.values()),
        marker=dict(
            color=['green', 'lightgreen', 'yellow', 'lightcoral', 'red'],
            line=dict(color='black', width=1)
        )
    )])
    fig_scenarios.update_layout(
        title="2020 Crime Projection Scenarios",
        yaxis_title="Estimated Crime Count",
        height=350
    )
    st.plotly_chart(fig_scenarios, use_container_width=True)
    
    # Crime-specific predictions
    st.subheader("🚨 Crime-Type Specific Predictions")
    
    selected_crime_pred = st.selectbox(
        "Select Crime Type for Individual Prediction:",
        crime_categories,
        key='crime_pred_select'
    )
    
    crime_ts = df[df['Offence'] == selected_crime_pred][['Year', 'Pakistan']].sort_values('Year')
    crime_years = crime_ts['Year'].values
    crime_values = crime_ts['Pakistan'].values
    
    if len(crime_values) >= 3:
        # Fit polynomial to crime data
        z_crime = np.polyfit(crime_years, crime_values, 2)
        p_crime = np.poly1d(z_crime)
        
        # Forecast for this crime
        crime_forecast = p_crime(future_years)
        
        # Calculate residuals for CI
        crime_residuals = crime_values - p_crime(crime_years)
        crime_std_error = np.std(crime_residuals)
        crime_ci = 1.96 * crime_std_error
        
        crime_upper = crime_forecast + crime_ci
        crime_lower = np.maximum(crime_forecast - crime_ci, 0)  # Can't be negative
        
        # Create crime-specific forecast plot
        fig_crime_pred = go.Figure()
        
        fig_crime_pred.add_trace(go.Scatter(
            x=crime_years,
            y=crime_values,
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        crime_forecast_full = np.concatenate([[crime_values[-1]], crime_forecast])
        crime_years_full = np.concatenate([[crime_years[-1]], future_years])
        
        fig_crime_pred.add_trace(go.Scatter(
            x=crime_years_full,
            y=crime_forecast_full,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=8)
        ))
        
        crime_upper_full = np.concatenate([[crime_values[-1]], crime_upper])
        crime_lower_full = np.concatenate([[crime_values[-1]], crime_lower])
        
        fig_crime_pred.add_trace(go.Scatter(
            x=crime_years_full,
            y=crime_upper_full,
            mode='lines',
            name='Upper CI',
            line=dict(color='rgba(255,0,0,0)')
        ))
        
        fig_crime_pred.add_trace(go.Scatter(
            x=crime_years_full,
            y=crime_lower_full,
            mode='lines',
            name='95% Confidence',
            line=dict(color='rgba(255,0,0,0)'),
            fillcolor='rgba(255,0,0,0.2)',
            fill='tonexty'
        ))
        
        fig_crime_pred.update_layout(
            title=f"{selected_crime_pred} Forecast 2018-2020",
            xaxis_title="Year",
            yaxis_title="Number of Cases",
            height=350
        )
        st.plotly_chart(fig_crime_pred, use_container_width=True)
        
        # Crime forecast details
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_val = crime_values[-1]
            forecast_2020 = crime_forecast[-1]
            change_pct = ((forecast_2020 - current_val) / current_val) * 100
            st.metric(
                "2020 Forecast",
                f"{int(forecast_2020):,}",
                f"{change_pct:+.1f}% from 2017"
            )
        
        with col2:
            st.metric(
                "Upper 95% CI",
                f"{int(crime_upper[-1]):,}",
                "Worst case estimate"
            )
        
        with col3:
            st.metric(
                "Lower 95% CI",
                f"{int(max(0, crime_lower[-1])):,}",
                "Best case estimate"
            )
        
        # Crime forecast table
        crime_forecast_df = pd.DataFrame({
            'Year': future_years,
            f'{selected_crime_pred}': crime_forecast.astype(int),
            'Upper CI': crime_upper.astype(int),
            'Lower CI': np.maximum(crime_lower, 0).astype(int),
            'Change %': ((crime_forecast - current_val) / current_val * 100).round(1)
        })
        
        st.dataframe(crime_forecast_df, use_container_width=True, hide_index=True)
    
    # Trend line
    trend_years = np.linspace(years_array.min(), years_array.max(), 100)
    fig_forecast = go.Figure()
    
    fig_forecast.add_trace(go.Scatter(
        x=years_array,
        y=crimes_array,
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))
    
    fig_forecast.add_trace(go.Scatter(
        x=trend_years,
        y=p_poly(trend_years),
        mode='lines',
        name='Trend Line',
        line=dict(color='green', width=2, dash='dash')
    ))
    
    # Forecast
    forecast_years_full = np.concatenate([[years_array[-1]], future_years])
    forecast_values_full = np.concatenate([[crimes_array[-1]], forecast_poly])
    
    fig_forecast.add_trace(go.Scatter(
        x=forecast_years_full,
        y=forecast_values_full,
        mode='lines+markers',
        name='Forecast (Projected)',
        line=dict(color='red', width=2, dash='dot'),
        marker=dict(size=8)
    ))
    
    fig_forecast.update_layout(
        title="Crime Trend with Forecast (2018-2020 Projection)",
        xaxis_title="Year",
        yaxis_title="Number of Crimes",
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Forecast details
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("2018 Forecast", f"{int(forecast_values[0]):,}")
    with col2:
        st.metric("2019 Forecast", f"{int(forecast_values[1]):,}")
    with col3:
        st.metric("2020 Forecast", f"{int(forecast_values[2]):,}")
    
    # Crime acceleration analysis
    st.subheader("🚀 Crime Rate Acceleration")
    
    yoy_changes = yearly_totals_full.pct_change() * 100
    acceleration = yoy_changes.diff()
    
    fig_accel = go.Figure()
    fig_accel.add_trace(go.Bar(
        x=acceleration.index,
        y=acceleration.values,
        marker_color=['red' if x < 0 else 'green' for x in acceleration.values],
        name='Acceleration'
    ))
    fig_accel.update_layout(
        title="Crime Growth Acceleration (Change in YoY %)",
        xaxis_title="Year",
        yaxis_title="Acceleration (%)",
        height=350
    )
    st.plotly_chart(fig_accel, use_container_width=True)

# ============================================================================
# TAB 5: DETAILED DATA
# ============================================================================
with tab5:
    st.subheader("Crime Analysis by Category")
    
    # Select specific crime for detailed analysis
    selected_crime_detail = st.selectbox(
        "🚨 Select Crime Type for Detailed Analysis:",
        crime_categories
    )
    
    crime_data_detail = df_filtered[df_filtered['Offence'] == selected_crime_detail].copy()
    
    if len(crime_data_detail) > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_crime = crime_data_detail['Pakistan'].sum()
            st.metric(f"Total {selected_crime_detail}", f"{int(total_crime):,}")
        
        with col2:
            avg_crime = crime_data_detail['Pakistan'].mean()
            st.metric("Annual Average", f"{int(avg_crime):,}")
        
        with col3:
            growth = ((crime_data_detail[crime_data_detail['Year'] == crime_data_detail['Year'].max()]['Pakistan'].values[0] -
                      crime_data_detail[crime_data_detail['Year'] == crime_data_detail['Year'].min()]['Pakistan'].values[0]) /
                     crime_data_detail[crime_data_detail['Year'] == crime_data_detail['Year'].min()]['Pakistan'].values[0]) * 100
            st.metric("Growth Rate", f"{growth:+.1f}%")
        
        # Time series for selected crime
        crime_yearly = crime_data_detail[['Year', 'Pakistan']].sort_values('Year')
        
        fig_crime_detail = go.Figure()
        fig_crime_detail.add_trace(go.Scatter(
            x=crime_yearly['Year'],
            y=crime_yearly['Pakistan'],
            mode='lines+markers',
            name=selected_crime_detail,
            fill='tozeroy',
            line=dict(width=3)
        ))
        fig_crime_detail.update_layout(
            title=f"{selected_crime_detail} Trend",
            xaxis_title="Year",
            yaxis_title="Cases",
            height=400
        )
        st.plotly_chart(fig_crime_detail, use_container_width=True)
        
        # Provincial breakdown
        st.subheader(f"Provincial Breakdown: {selected_crime_detail}")
        
        prov_breakdown = crime_data_detail[provinces_list].sum().sort_values(ascending=False)
        
        fig_prov_break = go.Figure(data=[go.Bar(
            x=prov_breakdown.index,
            y=prov_breakdown.values,
            marker_color=prov_breakdown.values,
            marker_colorscale='Viridis'
        )])
        fig_prov_break.update_layout(
            title=f"{selected_crime_detail} by Province",
            yaxis_title="Number of Cases",
            height=350
        )
        st.plotly_chart(fig_prov_break, use_container_width=True)

# ============================================================================
# TAB 6: DEEP STATISTICAL ANALYSIS
# ============================================================================
with tab6:
    st.subheader("Statistical & Correlation Analysis")
    
    # Prepare crime data for correlation
    crime_pivot = df_filtered[df_filtered['Offence'] != 'TOTAL RECORDED CRIME'].pivot_table(
        values='Pakistan',
        index='Year',
        columns='Offence',
        aggfunc='sum'
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Correlation heatmap
        st.subheader("Crime Type Correlations")
        corr_matrix = crime_pivot.corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 9},
            colorbar=dict(title="Correlation")
        ))
        fig_corr.update_layout(height=600, width=800)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        st.subheader("Key Correlations")
        corr_flat = corr_matrix.unstack()
        corr_flat = corr_flat[corr_flat != 1.0].sort_values(ascending=False)
        
        st.write("**Strongest Correlations:**")
        for idx, (pair, corr_val) in enumerate(corr_flat.head(10).items()):
            st.write(f"{idx+1}. {pair[0][:15]} ↔ {pair[1][:15]}: **{corr_val:.2f}**")
    
    # Distribution analysis
    st.subheader("Crime Distribution Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Skewness
        st.write("**Skewness (Distribution Shape):**")
        for crime in crime_pivot.columns[:5]:
            skew = crime_pivot[crime].skew()
            st.write(f"• {crime[:20]}: {skew:.2f}")
    
    with col2:
        # Kurtosis
        st.write("**Kurtosis (Peak Sharpness):**")
        for crime in crime_pivot.columns[:5]:
            kurt = crime_pivot[crime].kurtosis()
            st.write(f"• {crime[:20]}: {kurt:.2f}")
    
    with col3:
        # Coefficient of Variation
        st.write("**Variability (CV %):**")
        for crime in crime_pivot.columns[:5]:
            cv = (crime_pivot[crime].std() / crime_pivot[crime].mean()) * 100
            st.write(f"• {crime[:20]}: {cv:.1f}%")
    
    # Distribution plots
    st.subheader("Distribution Patterns")
    selected_crime_dist = st.selectbox("Select crime for distribution analysis:", crime_pivot.columns)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = go.Figure(data=[go.Histogram(
            x=crime_pivot[selected_crime_dist],
            nbinsx=10,
            marker_color='rgba(100, 100, 200, 0.7)'
        )])
        fig_hist.update_layout(title=f"Distribution: {selected_crime_dist}", height=350)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        fig_box = go.Figure(data=[go.Box(
            y=crime_pivot[selected_crime_dist],
            name=selected_crime_dist,
            marker_color='lightblue'
        )])
        fig_box.update_layout(title=f"Box Plot: {selected_crime_dist}", height=350)
        st.plotly_chart(fig_box, use_container_width=True)

# ============================================================================
# TAB 7: ADVANCED CRIME PATTERNS & INSIGHTS
# ============================================================================
with tab7:
    st.subheader("Advanced Crime Pattern Analysis")
    
    # Crime severity ranking
    st.subheader("🚨 Crime Severity Assessment")
    
    crime_severity = df_filtered[df_filtered['Offence'] != 'TOTAL RECORDED CRIME'].groupby('Offence')['Pakistan'].agg([
        ('Total Cases', 'sum'),
        ('Avg Annual', 'mean'),
        ('Max Year', 'max'),
        ('Min Year', 'min'),
        ('Volatility', 'std')
    ]).sort_values('Total Cases', ascending=False)
    
    # Create severity score
    crime_severity['Severity Score'] = (
        (crime_severity['Total Cases'] / crime_severity['Total Cases'].max() * 0.4) +
        (crime_severity['Volatility'] / crime_severity['Volatility'].max() * 0.3) +
        (crime_severity['Max Year'] / crime_severity['Max Year'].max() * 0.3)
    ) * 100
    
    crime_severity_sorted = crime_severity.sort_values('Severity Score', ascending=False)
    
    fig_severity = go.Figure(data=[go.Bar(
        x=crime_severity_sorted['Severity Score'],
        y=crime_severity_sorted.index,
        orientation='h',
        marker=dict(
            color=crime_severity_sorted['Severity Score'],
            colorscale='Reds',
            colorbar=dict(title="Severity")
        )
    )])
    fig_severity.update_layout(
        title="Crime Severity Score (Composite Index)",
        xaxis_title="Severity Score",
        height=500
    )
    st.plotly_chart(fig_severity, use_container_width=True)
    
    st.dataframe(crime_severity_sorted, use_container_width=True)
    
    # Provincial crime burden
    st.subheader("🗺️ Provincial Crime Burden Analysis")
    
    prov_data = df_filtered[df_filtered['Offence'] == 'TOTAL RECORDED CRIME'][provinces_list].iloc[0]
    total_national = prov_data.sum()
    
    prov_burden = pd.DataFrame({
        'Province': prov_data.index,
        'Crime Count': prov_data.values,
        'Percentage': (prov_data.values / total_national * 100)
    }).sort_values('Crime Count', ascending=False)
    
    fig_burden = go.Figure(data=[
        go.Bar(x=prov_burden['Province'], y=prov_burden['Percentage'], name='Crime %'),
    ])
    fig_burden.update_layout(
        title="Provincial Crime Burden (%)",
        yaxis_title="Percentage of National Crimes",
        height=350
    )
    st.plotly_chart(fig_burden, use_container_width=True)
    
    # Year-wise volatility
    st.subheader("📊 Crime Volatility Over Years")
    
    yearly_volatility = df_filtered[df_filtered['Offence'] != 'TOTAL RECORDED CRIME'].groupby('Year')['Pakistan'].std()
    
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=yearly_volatility.index,
        y=yearly_volatility.values,
        mode='lines+markers',
        name='Volatility (Std Dev)',
        line=dict(color='orange', width=3),
        marker=dict(size=10),
        fill='tozeroy'
    ))
    fig_vol.update_layout(
        title="Crime Volatility Trend",
        xaxis_title="Year",
        yaxis_title="Standard Deviation",
        height=350
    )
    st.plotly_chart(fig_vol, use_container_width=True)
    
    # Crime concentration
    st.subheader("🎯 Crime Concentration Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Herfindahl index (concentration)
        crime_totals = df_filtered[df_filtered['Offence'] != 'TOTAL RECORDED CRIME'].groupby('Offence')['Pakistan'].sum()
        total = crime_totals.sum()
        market_share = crime_totals / total
        herfindahl = (market_share ** 2).sum()
        
        st.metric("Herfindahl Index", f"{herfindahl:.3f}", 
                 help="Measures concentration of crime types. Higher = more concentrated")
        
        concentration_pct = crime_totals.head(5).sum() / total * 100
        st.metric("Top 5 Crimes Share", f"{concentration_pct:.1f}%",
                 help="% of crimes from top 5 categories")
    
    with col2:
        # Crime diversity
        unique_crimes = len(crime_totals)
        max_possible = np.log(unique_crimes)
        actual = -np.sum((crime_totals/total) * np.log(crime_totals/total + 1e-10))
        diversity_idx = actual / max_possible if max_possible > 0 else 0
        
        st.metric("Diversity Index", f"{diversity_idx:.2f}",
                 help="Higher = more diverse crime types")
        
        top_crime_pct = crime_totals.max() / total * 100
        st.metric("Dominant Crime %", f"{top_crime_pct:.1f}%",
                 help="% of total crimes from most common type")

# ============================================================================
# TAB 8: DATA EXPLORER WITH ADVANCED FILTERS
# ============================================================================
with tab8:
    st.subheader("Advanced Data Explorer")
    
    # Multi-filter interface
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_crime = st.selectbox(
            "Select Crime Type:",
            ['All'] + crime_categories,
            key='filter_crime_tab8'
        )
    
    with col2:
        filter_province = st.selectbox(
            "Select Province:",
            ['All', 'Pakistan'] + provinces_list,
            key='filter_province_tab8'
        )
    
    with col3:
        filter_year = st.selectbox(
            "Select Year:",
            ['All'] + sorted(df_filtered['Year'].unique()),
            key='filter_year_tab8'
        )
    
    # Apply filters
    display_df = df_filtered.copy()
    
    if filter_crime != 'All':
        display_df = display_df[display_df['Offence'] == filter_crime]
    
    if filter_year != 'All':
        display_df = display_df[display_df['Year'] == filter_year]
    
    if filter_province == 'All':
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        display_df = display_df[['_id', 'Year', 'Offence', filter_province]].rename(
            columns={filter_province: 'Crime Count'}
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Download data
    col1, col2 = st.columns(2)
    
    with col1:
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv,
            file_name="crime_data_filtered.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        json_data = display_df.to_json()
        st.download_button(
            label="📥 Download as JSON",
            data=json_data,
            file_name="crime_data_filtered.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Data insights
    st.subheader("Data Insights")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(display_df))
    with col2:
        st.metric("Total Cases", int(display_df['Pakistan'].sum()) if 'Pakistan' in display_df.columns else "N/A")
    with col3:
        st.metric("Years Covered", display_df['Year'].nunique())
    with col4:
        st.metric("Crime Types", display_df['Offence'].nunique())

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
    <div style='text-align: center; color: #888; font-size: 12px;'>
    <p>Pakistan Crime Analysis Dashboard | Data Period: 2012-2017</p>
    <p>Built with Streamlit | Data Source: Pakistan Crime Statistics</p>
    </div>
""", unsafe_allow_html=True)
