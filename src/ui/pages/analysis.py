"""Analysis configuration page."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, Optional

from core.nca_analyzer import NCAAnalyzer
from utils.constants import (
    ANALYSIS_METHODS,
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_CONFIDENCE_LEVEL
)
from utils.session_state import (
    validate_session_state,
    update_analysis_settings,
    validate_analysis_settings
)

def render_results(results: Dict[str, Any]) -> None:
    """Render analysis results."""
    st.header("Analysis Results")
    
    # Display main plot
    fig = go.Figure()
    
    # Get data from session state
    processed_data = st.session_state.processed_data
    x = processed_data[st.session_state.x_var].values
    y = processed_data[st.session_state.y_var].values
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name='Data Points',
        marker=dict(
            size=8,
            color='blue',
            opacity=0.6
        )
    ))
    
    # Add ceiling line
    sort_idx = np.argsort(x)
    fig.add_trace(go.Scatter(
        x=x[sort_idx],
        y=results['ceiling_line'][sort_idx],
        mode='lines',
        name='Ceiling Line',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title="NCA Results",
        xaxis_title=st.session_state.x_var,
        yaxis_title=st.session_state.y_var,
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display key metrics
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Effect Size",
            f"{results['effect_size']:.3f}",
            help="NCA effect size (0-1)"
        )
    
    with col2:
        st.metric(
            "Points Below Ceiling",
            f"{results['effect_stats']['points_below_ceiling']}",
            help="Number of points below the ceiling line"
        )
    
    with col3:
        st.metric(
            "Model Fit (R²)",
            f"{1 - results['statistical_tests']['residual_std']**2:.3f}",
            help="Goodness of fit measure"
        )
    
    # Statistical Tests
    if results.get('statistical_tests'):
        st.subheader("Statistical Tests")
        stats_df = pd.DataFrame({
            'Metric': ['Significance (p-value)', 'Confidence Interval (Lower)', 'Confidence Interval (Upper)'],
            'Value': [
                f"{results['statistical_tests'].get('p_value', 'N/A'):.3f}",
                f"{results['statistical_tests'].get('ci_lower', 'N/A'):.3f}",
                f"{results['statistical_tests'].get('ci_upper', 'N/A'):.3f}"
            ]
        })
        st.dataframe(stats_df, hide_index=True)
    
    # Diagnostic Plots
    if results.get('diagnostics'):
        st.subheader("Diagnostic Plots")
        # Add diagnostic plots here based on what's available in results
        pass
    
    # Download Results
    st.download_button(
        label="Download Results CSV",
        data=pd.DataFrame(results).to_csv(index=False),
        file_name="nca_results.csv",
        mime="text/csv"
    )

def render() -> None:
    """Render the analysis configuration page."""
    st.title("Analysis Configuration")
    
    # Validate prerequisites
    is_valid, message = validate_session_state()
    if not is_valid:
        st.error(message)
        if st.button("Go to Data Upload"):
            st.session_state.current_page = 'data_prep'
            st.rerun()
        return
    
    # Check if we have results to display
    if 'analysis_results' in st.session_state and st.session_state.analysis_complete:
        render_results(st.session_state.analysis_results)
        if st.button("Configure New Analysis", type="secondary"):
            st.session_state.analysis_complete = False
            st.session_state.analysis_results = None
            st.rerun()
        return
    
    # Prepare data if not already prepared
    if not hasattr(st.session_state.data_processor, 'processed_data') or st.session_state.data_processor.processed_data is None:
        try:
            processed_data, preprocessing_info = st.session_state.data_processor.prepare_data_for_analysis(
                st.session_state.x_var,
                st.session_state.y_var,
                st.session_state.preprocessing_settings
            )
            st.session_state.processed_data = processed_data
            st.session_state.preprocessing_info = preprocessing_info
        except Exception as e:
            st.error(f"Error preparing data: {str(e)}")
            return
    
    # Initialize analysis settings if needed
    if 'analysis_settings' not in st.session_state:
        initialize_analysis_settings()
    
    render_method_selection()
    render_statistical_settings()
    render_advanced_options()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Preview Analysis", type="secondary"):
            preview_analysis()
    
    with col2:
        if st.button("Run Full Analysis", type="primary"):
            run_full_analysis()

def initialize_analysis_settings() -> None:
    """Initialize default analysis settings."""
    st.session_state.analysis_settings = {
        'method': 'CE-FDH',
        'bootstrap_iterations': DEFAULT_BOOTSTRAP_ITERATIONS,
        'confidence_level': DEFAULT_CONFIDENCE_LEVEL,
        'cr_fdh_bandwidth': 0.1,
        'qr_quantile': 0.95,
        'run_permutation_test': True,
        'compute_effect_size_ci': True,
        'compute_diagnostics': True,
        'analyze_residuals': True
    }

def render_method_selection() -> None:
    """Render the method selection section."""
    st.header("1. Method Selection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        method = st.selectbox(
            "NCA Method",
            list(ANALYSIS_METHODS.keys()),
            help="Select the NCA analysis method",
            index=list(ANALYSIS_METHODS.keys()).index(
                st.session_state.analysis_settings.get('method', 'CE-FDH')
            )
        )
        st.session_state.analysis_settings['method'] = method
        
        # Show method description
        st.markdown(f"*{ANALYSIS_METHODS[method]['description']}*")
    
    with col2:
        if method == "CR-FDH":
            bandwidth = st.slider(
                "Bandwidth",
                min_value=0.01,
                max_value=0.5,
                value=st.session_state.analysis_settings.get('cr_fdh_bandwidth', 0.1),
                help="Controls smoothness of the ceiling line"
            )
            st.session_state.analysis_settings['cr_fdh_bandwidth'] = bandwidth
            
        elif method == "Quantile":
            quantile = st.slider(
                "Quantile",
                min_value=0.8,
                max_value=0.99,
                value=st.session_state.analysis_settings.get('qr_quantile', 0.95),
                help="Percentile for regression line"
            )
            st.session_state.analysis_settings['qr_quantile'] = quantile

def render_statistical_settings() -> None:
    """Render statistical analysis configuration."""
    st.header("2. Statistical Analysis Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        bootstrap_iterations = st.number_input(
            "Bootstrap Iterations",
            min_value=100,
            max_value=10000,
            value=st.session_state.analysis_settings.get(
                'bootstrap_iterations', 
                DEFAULT_BOOTSTRAP_ITERATIONS
            ),
            step=100,
            help="More iterations = more precise confidence intervals"
        )
        st.session_state.analysis_settings['bootstrap_iterations'] = bootstrap_iterations
    
    with col2:
        confidence_level = st.slider(
            "Confidence Level",
            min_value=0.8,
            max_value=0.99,
            value=st.session_state.analysis_settings.get(
                'confidence_level',
                DEFAULT_CONFIDENCE_LEVEL
            ),
            help="Confidence level for statistical tests"
        )
        st.session_state.analysis_settings['confidence_level'] = confidence_level

def render_advanced_options() -> None:
    """Render advanced statistical settings."""
    with st.expander("Advanced Statistical Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.analysis_settings['run_permutation_test'] = st.checkbox(
                "Permutation Tests",
                value=st.session_state.analysis_settings.get('run_permutation_test', True),
                help="Test statistical significance through permutation"
            )
            
            st.session_state.analysis_settings['compute_effect_size_ci'] = st.checkbox(
                "Effect Size CI",
                value=st.session_state.analysis_settings.get('compute_effect_size_ci', True),
                help="Calculate confidence intervals for effect size"
            )
        
        with col2:
            st.session_state.analysis_settings['compute_diagnostics'] = st.checkbox(
                "Diagnostic Metrics",
                value=st.session_state.analysis_settings.get('compute_diagnostics', True),
                help="Calculate model fit metrics"
            )
            
            st.session_state.analysis_settings['analyze_residuals'] = st.checkbox(
                "Residual Analysis",
                value=st.session_state.analysis_settings.get('analyze_residuals', True),
                help="Analyze residuals distribution"
            )

def preview_analysis() -> None:
    """Generate and display analysis preview."""
    with st.spinner("Calculating preview..."):
        try:
            processed_data = st.session_state.processed_data
            
            x = processed_data[st.session_state.x_var].values
            y = processed_data[st.session_state.y_var].values
            
            # Initialize analyzer
            analyzer = NCAAnalyzer(method=st.session_state.analysis_settings['method'])
            
            # Create method parameters
            method_params = {
                'bandwidth': st.session_state.analysis_settings.get('cr_fdh_bandwidth', 0.1),
                'quantile': st.session_state.analysis_settings.get('qr_quantile', 0.95),
                'bootstrap_iterations': 100  # Reduced for preview
            }
            
            # Run quick analysis
            results = analyzer.analyze(x, y, method_params)
            
            # Show preview plot
            fig = go.Figure()
            
            # Add scatter plot
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='markers',
                name='Data Points',
                marker=dict(
                    size=8,
                    color='blue',
                    opacity=0.6
                )
            ))
            
            # Add ceiling line
            sort_idx = np.argsort(x)
            fig.add_trace(go.Scatter(
                x=x[sort_idx],
                y=results['ceiling_line'][sort_idx],
                mode='lines',
                name='Ceiling Line',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title="NCA Preview",
                xaxis_title=st.session_state.x_var,
                yaxis_title=st.session_state.y_var,
                height=500,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show preview statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Preliminary Effect Size",
                    f"{results['effect_size']:.3f}"
                )
            
            with col2:
                st.metric(
                    "Points Below Ceiling",
                    f"{results['effect_stats']['points_below_ceiling']}"
                )
            
            with col3:
                st.metric(
                    "Model Fit (R²)",
                    f"{1 - results['statistical_tests']['residual_std']**2:.3f}"
                )
                
        except Exception as e:
            st.error(f"Error in analysis preview: {str(e)}")

def run_full_analysis() -> None:
    """Run complete analysis and store results."""
    # Validate settings
    is_valid, errors = validate_analysis_settings()
    if not is_valid:
        for error in errors:
            st.error(error)
        return
    
    with st.spinner("Running full analysis..."):
        try:
            # Get processed data
            processed_data = st.session_state.processed_data
            
            x = processed_data[st.session_state.x_var].values
            y = processed_data[st.session_state.y_var].values
            
            # Initialize analyzer
            analyzer = NCAAnalyzer(method=st.session_state.analysis_settings['method'])
            
            # Run analysis
            results = analyzer.analyze(x, y, st.session_state.analysis_settings)
            
            # Store results and mark analysis as complete
            st.session_state.analysis_results = results
            st.session_state.analysis_complete = True
            st.session_state.analysis_timestamp = pd.Timestamp.now()
            
            # Force page rerun to show results
            st.rerun()
            
        except Exception as e:
            st.error(f"Error in analysis: {str(e)}")
