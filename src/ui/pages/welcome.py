"""Welcome page for the NCA Analysis Tool."""
import streamlit as st
from utils.constants import CUSTOM_CSS

def render() -> None:
    """Render the welcome page with enhanced structure and error handling."""
    try:
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
        
        st.title("📊 Necessary Condition Analysis (NCA) Scientific Tool")
        
        st.markdown("""
        ## Welcome to the Advanced NCA Analysis Platform
        
        This tool provides comprehensive Necessary Condition Analysis capabilities for research 
        and scientific applications, following rigorous statistical methodologies.
        
        ### Key Features:
        - Multiple NCA methodologies (CE-FDH, CR-FDH, Quantile Regression)
        - Advanced statistical analysis and validation
        - Publication-ready visualizations
        - Comprehensive reporting capabilities
        - Scientific methodology documentation
        """)
        
        # Quick start guide with improved content
        with st.expander("📚 Quick Start Guide", expanded=True):
            st.markdown("""
            ### Data Preparation
            1. Prepare your CSV file:
               - Ensure variables are in columns
               - Remove unnecessary columns
               - Handle missing values
               - Minimum sample size: 30
            
            ### Analysis Options
            - **CE-FDH**: Traditional ceiling line approach
            - **CR-FDH**: Regression-based ceiling estimation
            - **Quantile Regression**: Alternative ceiling estimation
            
            ### Preprocessing Options
            - Outlier detection and removal
            - Data scaling and normalization
            - Missing value handling
            - Distribution transformation
            """)
        
        # Enhanced methodology section
        with st.expander("🔬 Methodology", expanded=False):
            render_methodology_section()
        
        # Start button with session state management
        if st.button("Start Analysis", type="primary"):
            st.session_state.current_page = 'data_prep'
            st.rerun()
            
    except Exception as e:
        st.error(f"Error in welcome page: {str(e)}")
        if st.button("Reset Application"):
            from utils.session_state import reset_session_state
            reset_session_state()
            st.rerun()

def render_methodology_section() -> None:
    """Render methodology content."""
    st.markdown("""
    ### Necessary Condition Analysis
    
    NCA is a method for identifying necessary (but not sufficient) conditions in data sets. 
    It helps researchers understand what minimum levels of X are required for specific levels of Y.
    
    #### Core Concepts:
    1. **Ceiling Line**: Upper boundary of the data space
    2. **Effect Size**: Constraint imposed by necessary condition
    3. **Bottleneck Analysis**: Required levels analysis
    
    #### Statistical Foundation:
    - Ceiling techniques (CE-FDH, CR-FDH)
    - Bootstrap validation
    - Significance testing
    """)

if __name__ == "__main__":
    render()