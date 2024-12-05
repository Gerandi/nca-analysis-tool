"""Results dashboard page."""
import streamlit as st
import numpy as np
import pandas as pd
import json
from typing import Dict, Any

from core.visualizer import NCAVisualizer
from utils.session_state import get_analysis_results, get_current_data

def render() -> None:
    """Render the results dashboard."""
    st.title("NCA Analysis Results")
    
    # Get results and validate
    results, message = get_analysis_results()
    if results is None:
        st.error(f"No analysis results available. {message}")
        if st.button("Go to Analysis"):
            st.session_state.current_page = 'analysis'
            st.rerun()
        return
    
    # Get current data
    data, data_message = get_current_data()
    if data is None:
        st.error(f"Data not available. {data_message}")
        return
        
    x = data[st.session_state.x_var].values
    y = data[st.session_state.y_var].values
    
    # Initialize visualizer
    visualizer = NCAVisualizer(results)
    
    # Render key metrics
    render_key_metrics(visualizer)
    
    # Main NCA plot
    st.header("NCA Plot")
    st.plotly_chart(
        visualizer.create_main_plot(x, y),
        use_container_width=True
    )
    
    # Diagnostic plots
    with st.expander("Diagnostic Plots", expanded=True):
        st.plotly_chart(
            visualizer.create_diagnostic_plots(x, y),
            use_container_width=True
        )
    
    # Bottleneck analysis
    st.header("Bottleneck Analysis")
    st.plotly_chart(
        visualizer.create_bottleneck_plot(),
        use_container_width=True
    )
    
    # Bottleneck table
    with st.expander("Bottleneck Table", expanded=False):
        st.dataframe(
            results['bottleneck_table'],
            use_container_width=True
        )
    
    # Statistical details
    with st.expander("Statistical Details", expanded=False):
        render_statistical_details(results)
    
    # Export options
    st.header("Export Options")
    render_export_options(results)

def render_key_metrics(visualizer: NCAVisualizer) -> None:
    """Render key metrics section."""
    st.header("Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    summary_cards = visualizer.create_summary_cards()
    
    with col1:
        st.metric(
            "Effect Size",
            summary_cards['effect_size']['value'],
            help=f"95% CI: {summary_cards['effect_size']['ci']}"
        )
    
    with col2:
        st.metric(
            "P-value",
            summary_cards['statistical_tests']['p_value']
        )
    
    with col3:
        st.metric(
            "Sample Size",
            summary_cards['data_summary']['n_obs']
        )
    
    with col4:
        st.metric(
            "Correlation",
            summary_cards['data_summary']['correlation']
        )

def render_statistical_details(results: Dict[str, Any]) -> None:
    """Render detailed statistical information."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Effect Size Statistics")
        st.write(f"Raw Effect: {results['effect_stats']['raw_effect']:.3f}")
        st.write(f"Effect Size: {results['effect_size']:.3f}")
        if 'bootstrap_results' in results:
            st.write(f"95% CI: [{results['bootstrap_results']['statistics']['effect_size_ci'][0]:.3f}, "
                    f"{results['bootstrap_results']['statistics']['effect_size_ci'][1]:.3f}]")
    
    with col2:
        st.subheader("Model Diagnostics")
        st.write(f"Residual Mean: {results['statistical_tests']['residual_mean']:.3f}")
        st.write(f"Residual Std: {results['statistical_tests']['residual_std']:.3f}")
        st.write(f"Normality Test p-value: {results['statistical_tests']['normality_test_p']:.3f}")

def render_export_options(results: Dict[str, Any]) -> None:
    """Render export options."""
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate Report"):
            st.session_state.current_page = 'export'
            st.rerun()
    
    with col2:
        download_results(results)

def download_results(results: Dict[str, Any]) -> None:
    """Prepare and trigger download of analysis results."""
    try:
        # Prepare results for download
        download_data = {
            'effect_size': float(results['effect_size']),
            'statistical_tests': results['statistical_tests'],
            'effect_stats': results['effect_stats'],
            'bottleneck_table': results['bottleneck_table'].to_dict(),
            'timestamp': pd.Timestamp.now().isoformat(),
            'analysis_settings': results['analysis_settings'],
            'data_summary': results['data_summary']
        }
        
        # Convert to JSON
        json_str = json.dumps(download_data, indent=2)
        
        # Create download button
        st.download_button(
            "Download Results (JSON)",
            data=json_str,
            file_name="nca_results.json",
            mime="application/json"
        )
        
    except Exception as e:
        st.error(f"Error preparing download: {str(e)}")

def format_value(value: float) -> str:
    """Format numeric value for display."""
    return f"{value:.3f}" if abs(value) < 1000 else f"{value:,.0f}"
