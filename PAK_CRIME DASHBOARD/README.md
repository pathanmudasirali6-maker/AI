# 📊 Pakistan Crime Analysis Dashboard

A comprehensive Streamlit-based interactive dashboard for deep analysis of Pakistan's crime statistics (2012-2017).

## 🚀 Features

### 📊 Overview Tab
- National crime overview with trend analysis
- Crime distribution pie charts
- All crime categories bar chart
- Key metrics summary

### 📍 Provincial Analysis Tab
- Provincial crime comparison
- Year-wise provincial trends
- Provincial crime heatmap
- Crime burden analysis by province

### 🚨 Crime Types Tab
- Detailed analysis of specific crime types
- Deep dive into: Murder, Kidnapping/Abduction, Robbery
- Crime type comparison over years
- Individual crime trend analysis

### 📉 Trends & Predictions Tab
- Year-over-year crime change analysis
- Statistical summary (mean, median, std deviation)
- Crime growth rate by category
- **Trend forecasting & projections (2018-2020)**
- **Advanced multi-model predictions**
- **Confidence interval analysis**

### 🔬 Deep Analysis Tab
- Crime type correlation analysis
- Distribution shape metrics (Skewness, Kurtosis)
- Variability analysis (Coefficient of Variation)
- Distribution patterns and box plots

### 📈 Statistical Insights Tab
- **Composite Crime Severity Score**
- Provincial crime burden analysis
- Crime volatility trends
- **Crime concentration analysis**
- **Crime diversity index**

### 🎯 Crime Patterns Tab
- Advanced pattern recognition
- Severity assessment
- Volatility analysis
- Concentration and diversity metrics

### 📋 Data Explorer Tab
- Advanced filtering by crime type, province, and year
- Interactive data table
- Download data as CSV or JSON
- Quick data insights

## 🎛️ Advanced Sidebar Features

### 📅 Time Period Settings
- Year range slider
- Quick presets (All Years, Last 3 Years, 2017 Only)

### 🗺️ Provincial Filters
- Multi-select provinces
- Quick buttons (All, Major Cities, Clear)

### 🚨 Crime Type Filters
- Quick category filters
- Violent Crimes preset
- Property Crimes preset

### 📊 Analytics Options
- Chart theme selection
- Toggle trend lines
- Show/hide statistics
- Data smoothing option

### 💡 Quick Insights
- Real-time metrics
- Top crime analysis
- Most affected province
- Period-over-period change

### 💾 Export Data
- Download filtered data (CSV)
- Download statistics (CSV)

### ❓ Help & Information
- Complete guide
- Data source information

## 🔄 Data Filtering

Real-time filters across multiple dimensions:
- **Time**: Year range selection
- **Geography**: Province selection
- **Crime Type**: Multiple crime category selection

## 📈 Key Analytical Features

### Crime Severity Scoring
Composite index combining:
- Total cases (40%)
- Volatility/Variability (30%)
- Peak year intensity (30%)

### Statistical Analysis
- Correlation heatmap between crime types
- Distribution analysis (Skewness, Kurtosis, CV)
- Time series analysis

### Forecasting
- Polynomial trend fitting
- 3-year crime projection
- Acceleration analysis
- Multiple model comparison
- 95% confidence intervals

## 🛠️ Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run crime.py
```

## 📦 Requirements

- pandas >= 2.0.0
- numpy >= 1.24.0
- plotly >= 5.14.0
- streamlit >= 1.28.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

## 📊 Data Source

Pakistan Crime Statistics Database (2012-2017)
- Provinces: Punjab, Sindh, KP, Balochistan, Islamabad, Railways, G.B, AJK
- Crime Categories: 9 major categories + Total recorded crime

---

**Version:** 2.0 Advanced | **Status:** Production Ready | **Last Updated:** January 2026
