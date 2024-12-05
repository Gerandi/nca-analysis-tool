"""Export and report generation page."""
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional

from core.report_generator import ReportGenerator
from utils.session_state import get_analysis_results, get_current_data
from utils.constants import CUSTOM_CSS

def render() -> None:
    """Render the export and report generation page."""
    st.title("Export Results & Generate Reports")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Validate prerequisites
    results, message = get_analysis_results()
    if results is None:
        st.error(f"No analysis results available. {message}")
        if st.button("Go to Analysis", type="primary"):
            st.session_state.current_page = 'analysis'
            st.rerun()
        return
    
    # Get current data
    data, data_message = get_current_data()
    if data is None:
        st.error(f"Data not available. {data_message}")
        return
    
    # Initialize report generator with error handling
    try:
        settings = getattr(st.session_state, 'analysis_settings', {})
        settings['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_gen = ReportGenerator(
            results=results,
            data=data,
            x_var=st.session_state.x_var,
            y_var=st.session_state.y_var,
            settings=settings
        )
    except Exception as e:
        st.error(f"Error initializing report generator: {str(e)}")
        st.info("Try rerunning the analysis with updated settings.")
        if st.button("Return to Analysis"):
            st.session_state.current_page = 'analysis'
            st.rerun()
        return
    
    # Layout with tabs for better organization
    tab_report, tab_export = st.tabs(["Generate Report", "Export Data"])
    
    with tab_report:
        render_report_options(report_gen)
    
    with tab_export:
        render_export_options(report_gen)

def render_report_options(report_gen: ReportGenerator) -> None:
    """Render enhanced report generation options."""
    st.header("Report Generation")
    
    # Report configuration
    with st.expander("Report Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            report_format = st.selectbox(
                "Report Format",
                ["PDF", "Excel", "LaTeX"],
                help="Select the format for your report"
            )
        
        with col2:
            include_sections = st.multiselect(
                "Include Sections",
                [
                    "Executive Summary",
                    "Methodology",
                    "Results",
                    "Statistical Details",
                    "Visualizations",
                    "Recommendations"
                ],
                default=[
                    "Executive Summary",
                    "Results",
                    "Statistical Details"
                ],
                help="Choose sections to include in the report"
            )
    
        # Format-specific options
        if report_format == "PDF":
            st.subheader("PDF Options")
            col1, col2 = st.columns(2)
            with col1:
                st.checkbox("Include Table of Contents", value=True, key="pdf_toc")
                st.checkbox("Include Page Numbers", value=True, key="pdf_pages")
            with col2:
                st.checkbox("Include Citations", value=True, key="pdf_citations")
                st.checkbox("Include References", value=True, key="pdf_refs")
                
            # Visual options
            st.subheader("Visual Elements")
            col1, col2 = st.columns(2)
            with col1:
                st.radio(
                    "Color Scheme",
                    ["Professional", "High Contrast", "Colorblind Friendly"],
                    key="pdf_colors"
                )
            with col2:
                st.radio(
                    "Font Size",
                    ["Small", "Medium", "Large"],
                    index=1,
                    key="pdf_font"
                )
                
        elif report_format == "Excel":
            st.subheader("Excel Options")
            col1, col2 = st.columns(2)
            with col1:
                st.checkbox("Include Raw Data", value=True, key="excel_raw")
                st.checkbox("Auto-filter Tables", value=True, key="excel_filter")
            with col2:
                st.checkbox("Include Charts", value=True, key="excel_charts")
                st.checkbox("Include Formulas", value=False, key="excel_formulas")
                
        elif report_format == "LaTeX":
            st.subheader("LaTeX Options")
            col1, col2 = st.columns(2)
            with col1:
                st.selectbox(
                    "Document Class",
                    ["article", "report", "paper"],
                    key="latex_class"
                )
                st.checkbox("Include Abstract", value=True, key="latex_abstract")
            with col2:
                st.selectbox(
                    "Bibliography Style",
                    ["plain", "apa", "ieee"],
                    key="latex_bib"
                )
                st.checkbox("Double Spacing", value=False, key="latex_spacing")
    
    # Generate report button with status
    if st.button("Generate Report", type="primary"):
        with st.spinner(f"Generating {report_format} report..."):
            try:
                # Get report generation options
                options = {
                    'format': report_format,
                    'sections': include_sections,
                    'settings': {
                        'pdf_toc': st.session_state.get('pdf_toc', True),
                        'pdf_pages': st.session_state.get('pdf_pages', True),
                        'pdf_citations': st.session_state.get('pdf_citations', True),
                        'pdf_refs': st.session_state.get('pdf_refs', True),
                        'pdf_colors': st.session_state.get('pdf_colors', 'Professional'),
                        'pdf_font': st.session_state.get('pdf_font', 'Medium'),
                        'excel_raw': st.session_state.get('excel_raw', True),
                        'excel_filter': st.session_state.get('excel_filter', True),
                        'excel_charts': st.session_state.get('excel_charts', True),
                        'excel_formulas': st.session_state.get('excel_formulas', False),
                        'latex_class': st.session_state.get('latex_class', 'article'),
                        'latex_abstract': st.session_state.get('latex_abstract', True),
                        'latex_bib': st.session_state.get('latex_bib', 'plain'),
                        'latex_spacing': st.session_state.get('latex_spacing', False)
                    }
                }
                
                if report_format == "PDF":
                    report_bytes = report_gen.generate_pdf_report(options)
                    mime_type = "application/pdf"
                    file_extension = "pdf"
                elif report_format == "Excel":
                    report_bytes = report_gen.generate_excel_report(options)
                    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    file_extension = "xlsx"
                else:  # LaTeX
                    report_bytes = report_gen.generate_latex_report(options).encode('utf-8')
                    mime_type = "text/plain"
                    file_extension = "tex"
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"nca_report_{timestamp}.{file_extension}"
                
                st.download_button(
                    "Download Report",
                    data=report_bytes,
                    file_name=filename,
                    mime=mime_type,
                    help=f"Download the generated {report_format} report"
                )
                
                st.success("Report generated successfully! Click above to download.")
                
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
                st.info("Please check your settings and try again.")

def render_export_options(report_gen: ReportGenerator) -> None:
    """Render enhanced data export options."""
    st.header("Data Export")
    
    with st.expander("Export Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.selectbox(
                "Export Format",
                ["CSV", "Excel", "JSON"],
                help="Select the format for data export"
            )
        
        with col2:
            export_content = st.multiselect(
                "Export Content",
                [
                    "Analysis Results",
                    "Raw Data",
                    "Processed Data",
                    "Statistical Tests",
                    "Bottleneck Analysis",
                    "Model Parameters"
                ],
                default=["Analysis Results", "Raw Data"],
                help="Choose what content to include in the export"
            )
        
        # Format-specific options
        if export_format == "Excel":
            st.subheader("Excel Export Options")
            col1, col2 = st.columns(2)
            with col1:
                st.checkbox("Separate Sheets", value=True, key="excel_sheets")
                st.checkbox("Include Metadata", value=True, key="excel_meta")
            with col2:
                st.checkbox("Format Tables", value=True, key="excel_format")
                st.checkbox("Include Charts", value=False, key="excel_viz")
                
        elif export_format == "CSV":
            st.subheader("CSV Export Options")
            col1, col2 = st.columns(2)
            with col1:
                st.selectbox(
                    "Delimiter",
                    [",", ";", "\\t"],
                    key="csv_delimiter"
                )
            with col2:
                st.checkbox("Include Headers", value=True, key="csv_headers")
                
        elif export_format == "JSON":
            st.subheader("JSON Export Options")
            col1, col2 = st.columns(2)
            with col1:
                st.checkbox("Pretty Print", value=True, key="json_pretty")
                st.checkbox("Include Metadata", value=True, key="json_meta")
            with col2:
                st.checkbox("Compact Arrays", value=False, key="json_compact")
                st.checkbox("Include Schema", value=False, key="json_schema")
    
    # Export button with status
    if st.button("Export Data", type="primary"):
        with st.spinner(f"Preparing {export_format} export..."):
            try:
                # Get export options
                options = {
                    'format': export_format,
                    'content': export_content,
                    'settings': {
                        'excel_sheets': st.session_state.get('excel_sheets', True),
                        'excel_meta': st.session_state.get('excel_meta', True),
                        'excel_format': st.session_state.get('excel_format', True),
                        'excel_viz': st.session_state.get('excel_viz', False),
                        'csv_delimiter': st.session_state.get('csv_delimiter', ','),
                        'csv_headers': st.session_state.get('csv_headers', True),
                        'json_pretty': st.session_state.get('json_pretty', True),
                        'json_meta': st.session_state.get('json_meta', True),
                        'json_compact': st.session_state.get('json_compact', False),
                        'json_schema': st.session_state.get('json_schema', False)
                    }
                }
                
                if export_format == "CSV":
                    export_bytes = report_gen.export_to_csv(export_content, options)
                    mime_type = "text/csv"
                    file_extension = "csv"
                elif export_format == "Excel":
                    export_bytes = report_gen.export_to_excel(export_content, options)
                    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    file_extension = "xlsx"
                else:  # JSON
                    export_bytes = report_gen.export_to_json(export_content, options)
                    mime_type = "application/json"
                    file_extension = "json"
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"nca_export_{timestamp}.{file_extension}"
                
                st.download_button(
                    "Download Export",
                    data=export_bytes,
                    file_name=filename,
                    mime=mime_type,
                    help=f"Download the exported {export_format} file"
                )
                
                st.success("Data exported successfully! Click above to download.")
                
            except Exception as e:
                st.error(f"Error exporting data: {str(e)}")
                st.info("Please check your export settings and try again.")
