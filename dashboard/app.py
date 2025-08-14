"""
Solar Fault Detection Dashboard
Interactive Streamlit application for monitoring solar panel performance
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from detect_faults import SolarFaultDetector
from physics_rules import SolarPhysicsRules

def load_data():
    """Load sample solar data"""
    try:
        data = pd.read_csv('data/sample_solar_data.csv')
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        return data
    except FileNotFoundError:
        st.error("Sample data not found. Please run the simulation first.")
        return None

def create_performance_overview(df):
    """Create performance overview metrics"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_efficiency = df['power'].sum() / df['theoretical_power'].sum() * 100
        st.metric("Average Efficiency", f"{avg_efficiency:.1f}%", 
                 help="Efficiency values >100% indicate synthetic data simulation scenarios designed to test fault detection algorithms")
    
    with col2:
        total_power = df['power'].sum() / 1000  # kWh
        st.metric("Total Energy", f"{total_power:.1f} kWh")
    
    with col3:
        fault_count = len(df[df['detected_fault'] != 'Normal'])
        st.metric("Fault Events", fault_count)
    
    with col4:
        critical_faults = len(df[df['fault_severity'] == 'Critical'])
        st.metric("Critical Issues", critical_faults, delta_color="inverse")

def create_enhanced_time_series(df):
    """Create enhanced time series with color-coded severity"""
    
    def get_severity_color(severity):
        """Return color based on severity level"""
        color_map = {
            'Low': '#28a745',    # Green
            'Medium': '#ffc107',  # Yellow  
            'High': '#fd7e14',    # Orange
            'Critical': '#dc3545' # Red
        }
        return color_map.get(severity, '#6c757d')
    
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Power Output', 'Efficiency Over Time', 'Fault Frequency', 'Energy Output'),
        vertical_spacing=0.08,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Power output with severity colors
    df_copy = df.copy()
    df_copy.loc[:, 'efficiency_pct'] = (df_copy['power'] / df_copy['theoretical_power'] * 100).clip(0, 100)
    
    # Color points by severity
    colors = df_copy['fault_severity'].map(get_severity_color)
    
    fig.add_trace(
        go.Scatter(
            x=df_copy['timestamp'], 
            y=df_copy['power'], 
            mode='markers+lines',
            marker=dict(color=colors, size=6),
            name='Actual Power',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_copy['timestamp'], 
            y=df_copy['theoretical_power'], 
            name='Theoretical Power',
            line=dict(color='green', dash='dash')
        ),
        row=1, col=1
    )
    
    # Efficiency over time
    fig.add_trace(
        go.Scatter(
            x=df_copy['timestamp'], 
            y=df_copy['efficiency_pct'], 
            mode='markers+lines',
            marker=dict(color=colors, size=4),
            name='Efficiency',
            line=dict(color='purple', width=1)
        ),
        row=2, col=1
    )
    
    # Fault frequency (rolling count)
    df_copy.loc[:, 'fault_occurred'] = df_copy['detected_fault'] != 'Normal'
    rolling_faults = df_copy['fault_occurred'].rolling(window=24).sum()  # 6-hour rolling
    
    fig.add_trace(
        go.Scatter(
            x=df_copy['timestamp'], 
            y=rolling_faults, 
            name='Fault Frequency (6h rolling)',
            line=dict(color='red', width=2),
            fill='tozeroy'
        ),
        row=3, col=1
    )
    
    # Energy output (cumulative)
    df_copy.loc[:, 'energy_actual'] = df_copy['power'].cumsum() / 1000  # kWh
    df_copy.loc[:, 'energy_theoretical'] = df_copy['theoretical_power'].cumsum() / 1000  # kWh
    
    fig.add_trace(
        go.Scatter(
            x=df_copy['timestamp'], 
            y=df_copy['energy_actual'], 
            name='Actual Energy',
            line=dict(color='blue', width=2)
        ),
        row=4, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_copy['timestamp'], 
            y=df_copy['energy_theoretical'], 
            name='Theoretical Energy',
            line=dict(color='green', dash='dash')
        ),
        row=4, col=1
    )
    
    fig.update_layout(height=1000, showlegend=True, title_text="Enhanced Solar Panel Analytics")
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Power (W)", row=1, col=1)
    fig.update_yaxes(title_text="Efficiency (%)", row=2, col=1)
    fig.update_yaxes(title_text="Fault Count", row=3, col=1)
    fig.update_yaxes(title_text="Energy (kWh)", row=4, col=1)
    
    return fig

def create_fault_distribution_plot(df):
    """Create interactive fault type distribution with color-coded severity"""
    
    fault_counts = df['detected_fault'].value_counts()
    
    # Enhanced color mapping for better UX
    color_map = {
        'Normal': '#28a745',
        'Shading': '#ffc107',
        'Soiling': '#8B4513',
        'Degradation': '#007bff',
        'Hotspot': '#dc3545',
        'Bypass_Diode_Failure': '#6f42c1',
        'Minor_Anomaly': '#fd7e14',
        'Performance_Degradation': '#17a2b8',
        'Partial_Shading': '#e83e8c'
    }
    
    fig = px.pie(
        values=fault_counts.values,
        names=fault_counts.index,
        title="Fault Type Distribution",
        color=fault_counts.index,
        color_discrete_map=color_map,
        hole=0.3  # Donut chart for better visual appeal
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    
    fig.update_layout(
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="right", x=1)
    )
    
    return fig

def create_efficiency_heatmap(df):
    """Create efficiency vs irradiance heatmap"""
    
    # Create bins for analysis
    df_copy = df.copy()
    df_copy.loc[:, 'efficiency_pct'] = (df_copy['power'] / df_copy['theoretical_power'] * 100).clip(0, 100)
    
    # Create pivot table for heatmap
    pivot_data = df_copy.pivot_table(
        values='power', 
        index=pd.cut(df_copy['irradiance'], bins=20),
        columns=pd.cut(df_copy['efficiency_pct'], bins=20),
        aggfunc='count'
    )
    
    fig = px.imshow(
        pivot_data.values,
        labels=dict(x="Efficiency (%)", y="Irradiance (W/m²)", color="Count"),
        aspect="auto"
    )
    fig.update_layout(title="Performance Distribution Heatmap")
    
    return fig

def create_alerts_panel(df):
    """Display color-coded critical alerts"""
    
    severity_groups = df[df['detected_fault'] != 'Normal'].groupby('fault_severity')
    
    if len(severity_groups) > 0:
        st.warning("🚨 System Alerts Dashboard")
        
        # Color-coded alert containers with darker backgrounds for better visibility
        for severity, group in severity_groups:
            count = len(group)
            
            if severity == 'Critical':
                color = "#dc3545"
                icon = "🔴"
                bg_color = "rgba(220, 53, 69, 0.2)"
                text_color = "#721c24"
            elif severity == 'High':
                color = "#fd7e14"
                icon = "🟠"
                bg_color = "rgba(253, 126, 20, 0.2)"
                text_color = "#852d12"
            elif severity == 'Medium':
                color = "#ffc107"
                icon = "🟡"
                bg_color = "rgba(255, 193, 7, 0.2)"
                text_color = "#856404"
            else:  # Low
                color = "#28a745"
                icon = "🟢"
                bg_color = "rgba(40, 167, 69, 0.2)"
                text_color = "#155724"
            
            st.markdown(f"""
            <div style="padding: 1rem; margin: 0.5rem 0; border-radius: 0.5rem; border-left: 4px solid {color}; background-color: {bg_color};">
                <h4 style="margin: 0; color: {text_color}; font-weight: bold;">{icon} {severity} Severity ({count} issues)</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Display individual faults for this severity with better contrast
            for _, fault in group.iterrows():
                power_loss = max(0, fault['theoretical_power'] - fault['power'])
                st.markdown(f"""
                <div style="margin-left: 1rem; padding: 0.5rem; border-left: 2px solid {color}; background-color: #f8f9fa; border-radius: 0.25rem; margin-bottom: 0.25rem;">
                    <strong style="color: {text_color};">{fault['detected_fault']}</strong> at {fault['timestamp']}<br>
                    <small style="color: #495057;">Efficiency: {fault['efficiency']:.1f}% | Power Loss: {power_loss:.1f}W</small>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.success("✅ All systems operating normally")

def create_fault_bar_chart(df):
    """Create interactive bar chart for fault type distribution"""
    
    fault_counts = df['detected_fault'].value_counts()
    
    # Enhanced color mapping for bar chart
    color_map = {
        'Normal': '#28a745',
        'Shading': '#ffc107',
        'Soiling': '#8B4513',
        'Degradation': '#007bff',
        'Hotspot': '#dc3545',
        'Bypass_Diode_Failure': '#6f42c1',
        'Minor_Anomaly': '#fd7e14',
        'Performance_Degradation': '#17a2b8',
        'Partial_Shading': '#e83e8c'
    }
    
    colors = [color_map.get(fault, '#6c757d') for fault in fault_counts.index]
    
    fig = px.bar(
        x=fault_counts.index,
        y=fault_counts.values,
        title="Fault Type Frequency Analysis",
        labels={'x': 'Fault Type', 'y': 'Count'},
        color=colors,
        color_discrete_map="identity"
    )
    
    fig.update_traces(
        texttemplate='%{y}',
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Percentage: %{text}<extra></extra>',
        text=[f'{(val/sum(fault_counts.values)*100):.1f}%' for val in fault_counts.values]
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_tickangle=-45
    )
    
    return fig

def create_power_loss_trend(df):
    """Create interactive power loss trend analysis"""
    
    df_copy = df.copy()
    df_copy.loc[:, 'power_loss'] = (df_copy['theoretical_power'] - df_copy['power']).clip(lower=0)
    df_copy.loc[:, 'efficiency_pct'] = np.where(
        df_copy['theoretical_power'] <= 0,
        np.nan,
        (df_copy['power'] / df_copy['theoretical_power']) * 100
    )
    
    # Group by date for trend analysis
    daily_stats = df_copy.groupby(df_copy['timestamp'].dt.date, observed=False).agg({
        'power_loss': 'sum',
        'efficiency_pct': 'mean',
        'theoretical_power': 'sum'
    }).reset_index()
    
    # Create subplots for trend analysis
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Daily Power Loss Trend', 'Daily Efficiency Trend'),
        vertical_spacing=0.15
    )
    
    # Power loss trend
    fig.add_trace(
        go.Scatter(
            x=daily_stats['timestamp'],
            y=daily_stats['power_loss'],
            mode='lines+markers',
            name='Daily Power Loss (W)',
            line=dict(color='#dc3545', width=3),
            marker=dict(size=8, color='#dc3545'),
            fill='tozeroy',
            fillcolor='rgba(220, 53, 69, 0.1)'
        ),
        row=1, col=1
    )
    
    # Efficiency trend
    fig.add_trace(
        go.Scatter(
            x=daily_stats['timestamp'],
            y=daily_stats['efficiency_pct'],
            mode='lines+markers',
            name='Daily Efficiency (%)',
            line=dict(color='#007bff', width=3),
            marker=dict(size=8, color='#007bff'),
            fill='tozeroy',
            fillcolor='rgba(0, 123, 255, 0.1)'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="Power Loss & Efficiency Trend Analysis"
    )
    
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Power Loss (W)", row=1, col=1)
    fig.update_yaxes(title_text="Efficiency (%)", row=2, col=1)
    
    return fig

def create_recovery_suggestions(df):
    """Create recovery suggestions based on detected faults"""
    
    suggestions = {
        'Hotspot': 'Immediate inspection required. Check for hot spots, verify bypass diodes, consider thermal imaging.',
        'Bypass_Diode_Failure': 'Replace failed bypass diodes. Test string voltage balance and check for open circuits.',
        'Shading': 'Identify and remove shading sources. Consider panel repositioning or vegetation trimming.',
        'Soiling': 'Schedule cleaning. Check for dust, bird droppings, or debris accumulation.',
        'Degradation': 'Perform IV curve analysis. Consider panel replacement if degradation >20%.',
        'Partial_Shading': 'Check for partial shading patterns. Verify bypass diode functionality.',
        'Performance_Degradation': 'Conduct comprehensive system inspection including connections and inverter.',
        'Minor_Anomaly': 'Monitor closely. Check for loose connections or measurement calibration.',
        'Normal': 'System operating normally. Continue regular monitoring.'
    }
    
    fault_counts = df[df['detected_fault'] != 'Normal']['detected_fault'].value_counts()
    
    if len(fault_counts) > 0:
        st.subheader("📋 Recovery Suggestions")
        
        for fault_type, count in fault_counts.items():
            st.info(f"**{fault_type}** ({count} occurrences): {suggestions.get(fault_type, 'Perform detailed system inspection.')}")
    else:
        st.success("✅ All systems operating normally")

def main():
    """Main dashboard application"""
    
    st.set_page_config(
        page_title="Solar Fault Detection Dashboard",
        page_icon="☀️",
        layout="wide"
    )
    
    # Header
    st.title("☀️ Solar Fault Detection Dashboard")
    st.markdown("### Powered by Physics-Informed Heuristics")
    st.markdown("Real-time monitoring and analysis of solar panel performance")
    
    # Load data
    data = load_data()
    
    if data is None:
        st.stop()
    
    # Sidebar controls
    st.sidebar.header("🔧 Dashboard Controls")
    
    # Date range filter with presets
    date_preset = st.sidebar.selectbox("Quick Select", ["Custom", "Last 7 Days", "Last 14 Days", "This Month", "Last 30 Days"])
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(data['timestamp'].min().date(), data['timestamp'].max().date()),
        min_value=data['timestamp'].min().date(),
        max_value=data['timestamp'].max().date()
    )
    
    # Apply date presets
    if date_preset != "Custom":
        max_date = data['timestamp'].max().date()
        if date_preset == "Last 7 Days":
            date_range = (max_date - pd.Timedelta(days=7), max_date)
        elif date_preset == "Last 14 Days":
            date_range = (max_date - pd.Timedelta(days=14), max_date)
        elif date_preset == "This Month":
            date_range = (max_date.replace(day=1), max_date)
        elif date_preset == "Last 30 Days":
            date_range = (max_date - pd.Timedelta(days=30), max_date)
    
    # Fault type filter
    fault_filter = st.sidebar.multiselect(
        "Filter by Fault Type",
        options=data['detected_fault'].unique(),
        default=data['detected_fault'].unique()
    )
    
    # Filter data
    mask = (
        (data['timestamp'].dt.date >= date_range[0]) &
        (data['timestamp'].dt.date <= date_range[1]) &
        (data['detected_fault'].isin(fault_filter))
    )
    filtered_data = data[mask]
    
    # Summary Cards with Enhanced Analytics
    st.subheader("📊 System Overview")
    
    # Calculate enhanced metrics
    fault_data = filtered_data[filtered_data['detected_fault'] != 'Normal'].copy()
    total_faults = len(fault_data)
    
    # Most frequent fault type
    if total_faults > 0:
        fault_counts = fault_data['detected_fault'].value_counts()
        most_frequent = fault_counts.index[0] if len(fault_counts) > 0 else "None"
    else:
        most_frequent = "None"
    
    # Enhanced calculations with proper handling
    filtered_data_copy = filtered_data.copy()
    filtered_data_copy.loc[:, 'power_loss'] = (filtered_data_copy['theoretical_power'] - filtered_data_copy['power']).clip(lower=0)
    
    # Handle zero/negative theoretical power gracefully
    filtered_data_copy.loc[:, 'efficiency_pct'] = np.where(
        filtered_data_copy['theoretical_power'] <= 0,
        np.nan,
        (filtered_data_copy['power'] / filtered_data_copy['theoretical_power']) * 100
    )
    
    # Average daily efficiency (handle NaN values)
    daily_efficiency = filtered_data_copy.groupby(filtered_data_copy['timestamp'].dt.date, observed=False)['efficiency_pct'].mean()
    avg_daily_eff = daily_efficiency.mean() if len(daily_efficiency) > 0 else 0
    
    # Peak power loss hour
    hourly_loss = filtered_data_copy.groupby(filtered_data_copy['timestamp'].dt.hour, observed=False)['power_loss'].sum()
    peak_loss_hour = hourly_loss.idxmax() if len(hourly_loss) > 0 else 0
    
    # Total faults by type
    fault_summary = fault_data['detected_fault'].value_counts().to_dict()
    
    # Create enhanced summary cards with color-coded indicators
    st.subheader("📊 Enhanced System Overview")
    
    # Calculate efficiency display value
    efficiency_display = f"{avg_daily_eff:.1f}%" if not np.isnan(avg_daily_eff) else "Sensor Offline"
    
    # Determine severity colors based on metrics
    fault_color = "#dc3545" if total_faults > 10 else "#ffc107" if total_faults > 0 else "#28a745"
    efficiency_color = "#28a745" if avg_daily_eff > 85 else "#ffc107" if avg_daily_eff > 70 else "#dc3545"
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="padding: 1rem; border-radius: 0.5rem; border-left: 4px solid {fault_color}; background-color: #f8f9fa;">
            <h4 style="margin: 0; color: {fault_color};">🔍 Most Frequent</h4>
            <h2 style="margin: 0.5rem 0; color: #212529;">{most_frequent}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #007bff; background-color: #f8f9fa;">
            <h4 style="margin: 0; color: #007bff;">⏰ Peak Loss Hour</h4>
            <h2 style="margin: 0.5rem 0; color: #212529;">{peak_loss_hour:02d}:00</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="padding: 1rem; border-radius: 0.5rem; border-left: 4px solid {efficiency_color}; background-color: #f8f9fa;">
            <h4 style="margin: 0; color: {efficiency_color};">📈 Avg Efficiency</h4>
            <h2 style="margin: 0.5rem 0; color: #212529;">{efficiency_display}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style="padding: 1rem; border-radius: 0.5rem; border-left: 4px solid {fault_color}; background-color: #f8f9fa;">
            <h4 style="margin: 0; color: {fault_color};">⚠️ Total Faults</h4>
            <h2 style="margin: 0.5rem 0; color: #212529;">{total_faults}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Fault type breakdown
    if fault_summary:
        st.subheader("📈 Fault Distribution")
        fault_df = pd.DataFrame(list(fault_summary.items()), columns=['Fault Type', 'Count'])
        st.bar_chart(fault_df.set_index('Fault Type'))
    
    # Enhanced visualizations with interactive charts
    st.subheader("📈 Interactive Performance Analytics")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Performance Overview", "Fault Analysis", "Trend Analysis"])
    
    with tab1:
        st.plotly_chart(create_enhanced_time_series(filtered_data), use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_fault_distribution_plot(filtered_data), use_container_width=True)
        with col2:
            st.plotly_chart(create_fault_bar_chart(filtered_data), use_container_width=True)
    
    with tab3:
        st.plotly_chart(create_power_loss_trend(filtered_data), use_container_width=True)
    
    # Recovery suggestions and alerts
    create_recovery_suggestions(filtered_data)
    create_alerts_panel(filtered_data)
    
    # Statistics section
    st.subheader("📊 Detailed Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Fault Type Distribution:**")
        st.dataframe(filtered_data['detected_fault'].value_counts())
    
    with col2:
        st.write("**Severity Distribution:**")
        st.dataframe(filtered_data['fault_severity'].value_counts())
    
    # Create tabbed alerts panel
    st.subheader("⚠️ Active Alerts")
    
    fault_data = filtered_data[filtered_data['detected_fault'] != 'Normal'].copy()
    
    if not fault_data.empty:
        # Add power loss column for display
        fault_data.loc[:, 'power_loss'] = (fault_data['theoretical_power'] - fault_data['power']).clip(lower=0)
        
        # Map fault types for display
        fault_data.loc[:, 'fault_type'] = fault_data['detected_fault']
        fault_data.loc[:, 'severity'] = fault_data['fault_severity']
        
        def get_severity_color(severity):
            color_map = {
                'Low': '#28a745',
                'Medium': '#ffc107',
                'High': '#fd7e14',
                'Critical': '#dc3545'
            }
            return color_map.get(severity, '#6c757d')
        
        # Create tabs for different alert views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 All Alerts", 
            "🔴 Critical", 
            "🟠 High Priority", 
            "🟡 Medium Priority", 
            "🟢 Low Priority"
        ])
        
        with tab1:
            st.markdown("### 📊 All Active Alerts")
            all_faults = fault_data.sort_values(['severity', 'timestamp'], ascending=[False, False])
            
            for _, fault in all_faults.head(10).iterrows():
                color = get_severity_color(fault['severity'])
                st.markdown(f"""
                <div style="padding: 0.8rem; margin: 0.3rem 0; border-radius: 0.5rem; 
                            border-left: 4px solid {color}; background-color: #343a40;">
                    <strong style="color: {color}; font-size: 1rem;">{fault['severity']}</strong> - 
                    <strong>{fault['fault_type']}</strong><br>
                    <small style="color: #adb5bd;">{fault['timestamp']}</small><br>
                    <span style="color: #f8f9fa;">Efficiency: {fault['efficiency']:.1f}% | Power Loss: {fault['power_loss']:.1f}W</span>
                </div>
                """, unsafe_allow_html=True)
                
            if len(all_faults) > 10:
                st.info(f"Showing 10 of {len(all_faults)} total alerts")
        
        with tab2:
            st.markdown("### 🔴 Critical Alerts")
            critical_faults = fault_data[fault_data['severity'] == 'Critical']
            
            if not critical_faults.empty:
                for _, fault in critical_faults.iterrows():
                    st.markdown(f"""
                    <div style="padding: 1rem; margin: 0.5rem 0; border-radius: 0.5rem; 
                                border-left: 4px solid #dc3545; background-color: #2d0a0a;">
                        <strong style="color: #dc3545; font-size: 1.1rem;">{fault['fault_type']}</strong><br>
                        <small style="color: #f8f9fa;">{fault['timestamp']}</small><br>
                        <span style="color: #f8f9fa;">Efficiency: {fault['efficiency']:.1f}% | Power Loss: {fault['power_loss']:.1f}W</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ No Critical alerts")
        
        with tab3:
            st.markdown("### 🟠 High Priority Alerts")
            high_faults = fault_data[fault_data['severity'] == 'High']
            
            if not high_faults.empty:
                for _, fault in high_faults.iterrows():
                    st.markdown(f"""
                    <div style="padding: 1rem; margin: 0.5rem 0; border-radius: 0.5rem; 
                                border-left: 4px solid #fd7e14; background-color: #3d1a0a;">
                        <strong style="color: #fd7e14; font-size: 1.1rem;">{fault['fault_type']}</strong><br>
                        <small style="color: #f8f9fa;">{fault['timestamp']}</small><br>
                        <span style="color: #f8f9fa;">Efficiency: {fault['efficiency']:.1f}% | Power Loss: {fault['power_loss']:.1f}W</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ No High Priority alerts")
        
        with tab4:
            st.markdown("### 🟡 Medium Priority Alerts")
            medium_faults = fault_data[fault_data['severity'] == 'Medium']
            
            if not medium_faults.empty:
                for _, fault in medium_faults.iterrows():
                    st.markdown(f"""
                    <div style="padding: 1rem; margin: 0.5rem 0; border-radius: 0.5rem; 
                                border-left: 4px solid #ffc107; background-color: #4d3d0a;">
                        <strong style="color: #ffc107; font-size: 1.1rem;">{fault['fault_type']}</strong><br>
                        <small style="color: #f8f9fa;">{fault['timestamp']}</small><br>
                        <span style="color: #f8f9fa;">Efficiency: {fault['efficiency']:.1f}% | Power Loss: {fault['power_loss']:.1f}W</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ No Medium Priority alerts")
        
        with tab5:
            st.markdown("### 🟢 Low Priority Alerts")
            low_faults = fault_data[fault_data['severity'] == 'Low']
            
            if not low_faults.empty:
                for _, fault in low_faults.iterrows():
                    st.markdown(f"""
                    <div style="padding: 1rem; margin: 0.5rem 0; border-radius: 0.5rem; 
                                border-left: 4px solid #28a745; background-color: #0a2d0a;">
                        <strong style="color: #28a745; font-size: 1.1rem;">{fault['fault_type']}</strong><br>
                        <small style="color: #f8f9fa;">{fault['timestamp']}</small><br>
                        <span style="color: #f8f9fa;">Efficiency: {fault['efficiency']:.1f}% | Power Loss: {fault['power_loss']:.1f}W</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ No Low Priority alerts")
    else:
        st.success("✅ All systems operating normally")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Powered by Physics-Informed Neural Networks (PINNs)**
    
    This dashboard uses physics-based rules combined with machine learning to detect solar panel faults.
    Inspired by SmartHelio's Autopilot and SolarGPT technologies.
    """)

if __name__ == "__main__":
    main()