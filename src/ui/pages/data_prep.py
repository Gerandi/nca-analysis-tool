"""
Data preparation and preprocessing page.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple

from core.data_processor import DataProcessor
from utils.constants import (
    SUPPORTED_FILE_TYPES,
    PREPROCESSING_OPTIONS
)
from utils.session_state import (
    update_preprocessing_settings,
    initialize_session_state,
    update_navigation_state
)

def render() -> None:
    """
    Render the data preparation and preprocessing page.
    """
    st.title("Data Upload & Preprocessing")

    # Ensure session state is initialized
    initialize_session_state()

    # Ensure data processor is initialized
    if 'data_processor' not in st.session_state or st.session_state.data_processor is None:
        st.session_state.data_processor = DataProcessor()

    # File upload section
    render_file_upload_section()

    # Continue only if data is loaded
    if st.session_state.get('data') is not None:
        render_data_overview()
        render_variable_selection()

        if st.session_state.get('x_var') and st.session_state.get('y_var'):
            render_preprocessing_options()
            render_continue_button()

def render_file_upload_section() -> None:
    """
    Render the file upload section.
    """
    st.header("1. Data Upload")

    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=SUPPORTED_FILE_TYPES,
        help="Upload a CSV file containing your data for analysis"
    )

    if uploaded_file is not None:
        with st.spinner("Loading and validating data..."):
            df, message = st.session_state.data_processor.load_data(uploaded_file)

            if df is not None:
                st.success(message)
                st.session_state.data = df
            else:
                st.error(message)

def render_data_overview() -> None:
    """
    Render the data overview section.
    """
    st.header("2. Data Overview")

    tab1, tab2, tab3 = st.tabs(["Preview", "Summary Statistics", "Data Quality"])

    with tab1:
        st.dataframe(
            st.session_state.data.head(),
            use_container_width=True,
            height=200
        )
        st.caption(f"Showing first 5 rows of {len(st.session_state.data)} total rows")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", len(st.session_state.data))
            st.metric("Total Columns", len(st.session_state.data.columns))
        with col2:
            st.metric("Missing Values", st.session_state.data.isnull().sum().sum())
            st.metric(
                "Numeric Columns",
                len(st.session_state.data.select_dtypes(include=['int64', 'float64']).columns)
            )

    with tab3:
        render_data_quality_checks()

def render_data_quality_checks() -> None:
    """
    Render data quality checks.
    """
    quality_issues = []
    data = st.session_state.data

    # Check for missing values
    missing_cols = data.columns[data.isnull().any()].tolist()
    if missing_cols:
        quality_issues.append(f"Missing values in columns: {', '.join(missing_cols)}")

    # Check for constant columns
    constant_cols = [col for col in data.columns if data[col].nunique() == 1]
    if constant_cols:
        quality_issues.append(f"Constant columns: {', '.join(constant_cols)}")

    if quality_issues:
        st.warning("Data Quality Issues Found:")
        for issue in quality_issues:
            st.write(f"- {issue}")
    else:
        st.success("No major data quality issues found")

def render_variable_selection() -> None:
    """
    Render the variable selection section.
    """
    st.header("3. Variable Selection")

    data = st.session_state.data
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns

    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox(
            "Select Independent Variable (X)",
            numeric_cols,
            help="Variable that might be necessary for Y",
            key="x_var_select"
        )

    with col2:
        y_var = st.selectbox(
            "Select Dependent Variable (Y)",
            [col for col in numeric_cols if col != x_var],
            help="Outcome variable",
            key="y_var_select"
        )

    if x_var and y_var:
        st.session_state.x_var = x_var
        st.session_state.y_var = y_var
        render_initial_visualizations(x_var, y_var)

def render_initial_visualizations(x_var: str, y_var: str) -> None:
    """
    Render initial data visualizations.
    """
    st.header("4. Initial Data Visualization")

    data = st.session_state.data

    # Scatter plot
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=data[x_var],
        y=data[y_var],
        mode='markers',
        name='Data Points',
        marker=dict(
            size=8,
            color='blue',
            opacity=0.6,
            line=dict(color='white', width=1)
        )
    ))

    fig_scatter.update_layout(
        title=dict(text='Data Preview Plot', x=0.5, font=dict(size=20)),
        xaxis_title=x_var,
        yaxis_title=y_var,
        template='plotly_white',
        height=500
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

def render_preprocessing_visualization(data: pd.DataFrame, var: str, method_type: str, 
                                    method: str, settings: dict = None) -> go.Figure:
    """Create before/after visualization for preprocessing steps."""
    processor = DataProcessor()
    original_data = data[var].copy()
    
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Original Data', 'Processed Data'),
                        horizontal_spacing=0.1)
    
    # Original data histogram and box plot
    fig.add_trace(
        go.Histogram(x=original_data, name="Original",
                    nbinsx=30, opacity=0.7,
                    marker_color='blue'),
        row=1, col=1
    )
    fig.add_trace(
        go.Box(y=original_data, name="Original",
               boxpoints='outliers', jitter=0.3,
               marker_color='blue'),
        row=1, col=1
    )
    
    # Process data based on method type
    try:
        if method_type == 'outlier':
            processed_mask, _ = processor.detect_outliers(
                original_data, 
                method=method,
                threshold=settings.get('outlier_threshold', 3.0)
            )
            processed_data = original_data[~processed_mask]
        elif method_type == 'scaling':
            processed_data, _ = processor.scale_data(original_data, method=method)
        elif method_type == 'transform':
            processed_data, _ = processor.transform_data(original_data, method=method)
        else:
            processed_data = original_data
            
        # Processed data histogram and box plot
        fig.add_trace(
            go.Histogram(x=processed_data, name="Processed",
                        nbinsx=30, opacity=0.7,
                        marker_color='green'),
            row=1, col=2
        )
        fig.add_trace(
            go.Box(y=processed_data, name="Processed",
                   boxpoints='outliers', jitter=0.3,
                   marker_color='green'),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=400,
            title=f"{method_type.capitalize()} Method: {method}",
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    except Exception as e:
        st.error(f"Error generating visualization: {str(e)}")
        return None

def render_preprocessing_options() -> None:
    """
    Render preprocessing options.
    """
    st.header("5. Preprocessing Options")

    # Store current variable selection for visualization
    current_var = st.radio(
        "Select variable to visualize",
        [st.session_state.x_var, st.session_state.y_var],
        horizontal=True
    )

    with st.expander("Outlier Detection and Removal", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            outlier_method = st.selectbox(
                "Outlier Detection Method",
                PREPROCESSING_OPTIONS['outlier_methods'],
                help="Choose method for detecting outliers",
                key="outlier_method"
            )

        with col2:
            if outlier_method != 'none':
                outlier_threshold = st.slider(
                    "Outlier Threshold",
                    min_value=1.0,
                    max_value=5.0,
                    value=3.0,
                    step=0.1,
                    help="Threshold for outlier detection",
                    key="outlier_threshold"
                )
        
        if outlier_method != 'none':
            fig = render_preprocessing_visualization(
                st.session_state.data,
                current_var,
                'outlier',
                outlier_method,
                {'outlier_threshold': outlier_threshold}
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    with st.expander("Scaling Options"):
        scaling_method = st.selectbox(
            "Scaling Method",
            PREPROCESSING_OPTIONS['scaling_methods'],
            help="Choose method for scaling variables",
            key="scaling_method"
        )
        if scaling_method != 'none':
            fig = render_preprocessing_visualization(
                st.session_state.data,
                current_var,
                'scaling',
                scaling_method
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    with st.expander("Transformation Options"):
        transform_method = st.selectbox(
            "Transformation Method",
            PREPROCESSING_OPTIONS['transform_methods'],
            help="Choose method for transforming variables",
            key="transform_method"
        )
        if transform_method != 'none':
            fig = render_preprocessing_visualization(
                st.session_state.data,
                current_var,
                'transform',
                transform_method
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)

def render_continue_button() -> None:
    """
    Render the continue button and save settings.
    """
    container = st.container()
    col1, col2 = container.columns([3, 1])

    with col1:
        current_settings = []
        if st.session_state.get('outlier_method') != 'none':
            current_settings.append(f"✓ Outlier detection: {st.session_state.get('outlier_method')}")
        if st.session_state.get('scaling_method') != 'none':
            current_settings.append(f"✓ Scaling: {st.session_state.get('scaling_method')}")
        if st.session_state.get('transform_method') != 'none':
            current_settings.append(f"✓ Transformation: {st.session_state.get('transform_method')}")

        if current_settings:
            st.info("Current preprocessing settings:\n" + "\n".join(current_settings))
        else:
            st.info("No preprocessing methods selected (using raw data)")

    with col2:
        if st.button("Continue to Analysis", type="primary", key="continue_to_analysis"):
            try:
                # Update preprocessing settings
                update_preprocessing_settings({
                    'outlier_method': st.session_state.get('outlier_method', 'none'),
                    'outlier_threshold': st.session_state.get('outlier_threshold', 3.0),
                    'scaling_method': st.session_state.get('scaling_method', 'none'),
                    'transform_method': st.session_state.get('transform_method', 'none')
                })
                
                # Mark data as prepared
                st.session_state.data_prepared = True
                
                # Update current page
                st.session_state.current_page = 'analysis'
                
                # Force page rerun
                st.rerun()
                
            except Exception as e:
                st.error(f"Error saving preprocessing settings: {str(e)}")
