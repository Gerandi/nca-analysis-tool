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

def render_debug_info():
    """Debug function to show session state."""
    with st.expander("Debug Info", expanded=False):
        st.write("Session State:")
        st.json({
            'analysis_complete': bool(st.session_state.get('analysis_complete')),
            'has_results': 'analysis_results' in st.session_state,
            'current_page': st.session_state.get('current_page')
        })

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
    
    # Add confidence intervals if available
    if 'bootstrap_results' in results and 'statistics' in results['bootstrap_results']:
        ci_lower = results['bootstrap_results']['statistics']['ceiling_ci_lower']
        ci_upper = results['bootstrap_results']['statistics']['ceiling_ci_upper']
        
        fig.add_trace(go.Scatter(
            x=x[sort_idx],
            y=ci_upper[sort_idx],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=x[sort_idx],
            y=ci_lower[sort_idx],
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(255, 0, 0, 0.2)',
            fill='tonexty',
            name='95% CI'
        ))
    
    fig.update_layout(
        title=f"NCA Results ({results['method_stats']['method']})",
        xaxis_title=st.session_state.x_var,
        yaxis_title=st.session_state.y_var,
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display key metrics
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Effect Size",
            f"{results['effect_size']:.3f}",
            help="NCA effect size (0-1)"
        )
    
    with col2:
        st.metric(
            "Points Below Ceiling",
            results['effect_stats']['points_below_ceiling'],
            help="Number of points below the ceiling line"
        )
    
    with col3:
        st.metric(
            "Model Fit (R²)",
            f"{1 - float(results['statistical_tests']['residual_std'])**2:.3f}",
            help="Goodness of fit measure"
        )
    
    with col4:
        st.metric(
            "Sample Size",
            results['data_summary']['n_observations'],
            help="Number of observations"
        )

# Statistical Tests
    if results.get('statistical_tests'):
        st.subheader("Statistical Tests")
        stats_df = pd.DataFrame({
            'Metric': [
                'Permutation Test (p-value)',
                'Residual Normality Test (p-value)',
                'Residual Mean',
                'Residual Std',
                'Skewness',
                'Kurtosis'
            ],
            'Value': [
                format_test_value(results['statistical_tests'].get('permutation_p_value')),
                format_test_value(results['statistical_tests'].get('normality_test_p')),
                format_test_value(results['statistical_tests'].get('residual_mean')),
                format_test_value(results['statistical_tests'].get('residual_std')),
                format_test_value(results['statistical_tests'].get('residual_skewness')),
                format_test_value(results['statistical_tests'].get('residual_kurtosis'))
            ]
        })
        st.dataframe(stats_df, hide_index=True)
    
    # Bootstrap Results
    if 'bootstrap_results' in results and 'statistics' in results['bootstrap_results']:
        st.subheader("Bootstrap Analysis")
        boot_stats = results['bootstrap_results']['statistics']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Mean Effect Size (Bootstrap)",
                f"{boot_stats['effect_size_mean']:.3f}",
                help="Average effect size from bootstrap samples"
            )
        
        with col2:
            ci_lower, ci_upper = boot_stats['effect_size_ci']
            st.metric(
                "Effect Size 95% CI",
                f"{ci_lower:.3f} - {ci_upper:.3f}",
                help="95% confidence interval for effect size"
            )
        
        # Bootstrap distribution plot
        fig_boot = go.Figure()
        fig_boot.add_trace(go.Histogram(
            x=results['bootstrap_results']['effect_sizes'],
            nbinsx=30,
            name='Bootstrap Distribution'
        ))
        
        fig_boot.update_layout(
            title="Bootstrap Distribution of Effect Size",
            xaxis_title="Effect Size",
            yaxis_title="Frequency",
            showlegend=False,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_boot, use_container_width=True)
    
    # Bottleneck Table
    if 'bottleneck_table' in results:
        st.subheader("Bottleneck Analysis")
        st.markdown("""This table shows the required X levels for different Y performance levels. 
                   Bottleneck points indicate where X becomes a constraint on Y.""")
        
        bottleneck_df = results['bottleneck_table']
        
        # Format the table for display
        display_cols = [
            'Y Level (%)', 'Required X Level', 'Ceiling Y Level', 
            'X Level (Standardized)', 'Y Level (Standardized)', 'Is Bottleneck'
        ]
        formatted_df = bottleneck_df[display_cols].copy()
        formatted_df = formatted_df.round(3)
        formatted_df['Is Bottleneck'] = formatted_df['Is Bottleneck'].map({True: '✓', False: ''})
        
        st.dataframe(
            formatted_df,
            hide_index=True,
            use_container_width=True
        )
    
    if st.button("Download Results"):
        try:
            # Create summary data
            download_data = pd.DataFrame({
                'X Variable': [st.session_state.x_var],
                'Y Variable': [st.session_state.y_var],
                'Method': [results['method_stats']['method']],
                'Effect Size': [results['effect_size']],
                'Points Below Ceiling': [results['effect_stats']['points_below_ceiling']],
                'Sample Size': [results['data_summary']['n_observations']]
            })
            
            # Add bootstrap results if available
            if 'bootstrap_results' in results and 'statistics' in results['bootstrap_results']:
                boot_stats = results['bootstrap_results']['statistics']
                download_data['Bootstrap Mean Effect Size'] = boot_stats['effect_size_mean']
                download_data['Effect Size CI Lower'] = boot_stats['effect_size_ci'][0]
                download_data['Effect Size CI Upper'] = boot_stats['effect_size_ci'][1]
            
            csv = download_data.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="Download Full Results CSV",
                data=csv,
                file_name="nca_results.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error preparing download: {str(e)}")

def format_test_value(value):
    """Helper function to format test values."""
    if value is None:
        return 'N/A'
    try:
        return f"{float(value):.3f}"
    except:
        return str(value)

def render() -> None:
    """Render the analysis configuration page."""
    st.title("Analysis Configuration")
    
    # Check if we have results to display
    if hasattr(st.session_state, 'analysis_complete') and st.session_state.analysis_complete:
        render_results(st.session_state.analysis_results)
        if st.button("Configure New Analysis", type="secondary"):
            st.session_state.analysis_complete = False
            st.session_state.analysis_results = None
            st.rerun()
        return
    
    # Validate prerequisites
    is_valid, message = validate_session_state()
    if not is_valid:
        st.error(message)
        if st.button("Go to Data Upload"):
            st.session_state.current_page = 'data_prep'
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
    # Add debug information
    render_debug_info()
    
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
            st.session_state['analysis_results'] = results
            st.session_state['analysis_complete'] = True
            st.session_state['analysis_timestamp'] = pd.Timestamp.now()
            
            # Force page rerun to show results
            st.experimental_rerun()
            
        except Exception as e:
            st.error(f"Error in analysis: {str(e)}")
            st.exception(e)  # This will show the full traceback