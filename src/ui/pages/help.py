"""Help and documentation page."""
import streamlit as st
from utils.constants import CUSTOM_CSS

def render() -> None:
    """Render the help and documentation page."""
    st.title("Documentation & Help")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Quick navigation
    render_quick_nav()
    
    # Main content sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Getting Started",
        "Analysis Methods",
        "FAQs",
        "Troubleshooting"
    ])
    
    with tab1:
        render_getting_started()
    
    with tab2:
        render_analysis_methods()
    
    with tab3:
        render_faqs()
    
    with tab4:
        render_troubleshooting()

def render_quick_nav() -> None:
    """Render quick navigation section."""
    st.sidebar.header("Quick Navigation")
    
    sections = {
        "Getting Started": [
            "Data Requirements",
            "File Upload",
            "Variable Selection",
            "Preprocessing"
        ],
        "Analysis Methods": [
            "CE-FDH Method",
            "CR-FDH Method",
            "Quantile Regression",
            "Statistical Tests"
        ],
        "FAQs": [
            "Common Questions",
            "Best Practices",
            "Interpretation Guide"
        ],
        "Troubleshooting": [
            "Common Issues",
            "Error Messages",
            "Data Problems"
        ]
    }
    
    for section, subsections in sections.items():
        st.sidebar.subheader(section)
        for subsection in subsections:
            st.sidebar.markdown(f"- [{subsection}](#{subsection.lower().replace(' ', '-')})")

def render_getting_started() -> None:
    """Render getting started section."""
    st.header("Getting Started")
    
    # Data Requirements
    st.subheader("Data Requirements")
    st.markdown("""
    Before using the NCA Analysis Tool, ensure your data meets these requirements:
    
    1. **File Format**
        - CSV file format
        - Clear column headers
        - No special characters in column names
    
    2. **Data Structure**
        - Minimum sample size: 30 observations
        - At least 2 numeric columns
        - No more than 20% missing values
    
    3. **Variable Types**
        - Independent variable (X): Numeric
        - Dependent variable (Y): Numeric
        - Optional categorical variables for grouping
    """)
    
    # File Upload
    st.subheader("File Upload")
    st.markdown("""
    To upload your data:
    
    1. Prepare your CSV file according to requirements
    2. Navigate to "Data Upload & Preprocessing"
    3. Use the file uploader to select your file
    4. Wait for validation and confirmation
    
    The tool will automatically validate your data and provide feedback on any issues.
    """)
    
    # Variable Selection
    st.subheader("Variable Selection")
    st.markdown("""
    After uploading your data:
    
    1. Select your Independent Variable (X)
        - Variable that might be necessary for Y
        - Should be measurable and controllable
    
    2. Select your Dependent Variable (Y)
        - Outcome variable of interest
        - Should vary with X
    
    The tool will show preview visualizations to help you confirm your selection.
    """)
    
    # Preprocessing
    st.subheader("Preprocessing")
    st.markdown("""
    Available preprocessing options:
    
    1. **Outlier Detection**
        - Z-score method
        - IQR method
        - Isolation Forest method
    
    2. **Scaling**
        - Standard scaling
        - Robust scaling
        - Min-Max scaling
    
    3. **Transformations**
        - Log transformation
        - Square root transformation
        - Box-Cox transformation
    """)

def render_analysis_methods() -> None:
    """Render analysis methods section."""
    st.header("Analysis Methods")
    
    # CE-FDH Method
    st.subheader("CE-FDH Method")
    st.markdown("""
    The Ceiling Envelopment with Free Disposal Hull (CE-FDH) method:
    
    - Traditional ceiling line approach
    - Non-parametric method
    - Identifies the empirical ceiling line
    - Suitable for most applications
    
    **When to use:**
    - Default method for NCA
    - When data points clearly define an upper boundary
    - When you want a conservative estimate
    """)
    
    # CR-FDH Method
    st.subheader("CR-FDH Method")
    st.markdown("""
    The Ceiling Regression with Free Disposal Hull (CR-FDH) method:
    
    - Regression-based ceiling estimation
    - Uses local regression techniques
    - Smoother ceiling line than CE-FDH
    - Bandwidth parameter controls smoothness
    
    **When to use:**
    - When you want a smoother ceiling line
    - With larger datasets
    - When CE-FDH appears too jagged
    """)
    
    # Quantile Regression
    st.subheader("Quantile Regression")
    st.markdown("""
    The Quantile Regression method:
    
    - Alternative ceiling approach
    - Uses specified quantile (e.g., 95th)
    - Provides a parametric ceiling line
    - More robust to outliers
    
    **When to use:**
    - When you want a parametric approach
    - With noisy data
    - When traditional methods are too sensitive
    """)
    
    # Statistical Tests
    st.subheader("Statistical Tests")
    st.markdown("""
    Available statistical validations:
    
    1. **Effect Size**
        - Measures constraint strength
        - Bootstrap confidence intervals
        - Significance testing
    
    2. **Model Diagnostics**
        - Residual analysis
        - Normality tests
        - QQ plots
    
    3. **Bootstrapping**
        - Confidence intervals
        - Stability analysis
        - Permutation tests
    """)

def render_faqs() -> None:
    """Render frequently asked questions section."""
    st.header("Frequently Asked Questions")
    
    faqs = {
        "What is NCA?": """
        Necessary Condition Analysis (NCA) is a method that identifies necessary (but not sufficient) 
        conditions in data sets. It helps understand what minimum levels of X are required for 
        specific levels of Y.
        """,
        
        "How do I interpret the effect size?": """
        - 0.0 - 0.1: Very small effect
        - 0.1 - 0.3: Small effect
        - 0.3 - 0.5: Medium effect
        - > 0.5: Large effect
        
        The effect size represents the constraint the necessary condition imposes on the outcome.
        """,
        
        "Which analysis method should I choose?": """
        - CE-FDH: Default choice, good for most cases
        - CR-FDH: When you want smoother ceiling lines
        - Quantile: When dealing with noisy data or outliers
        
        Start with CE-FDH and explore others if needed.
        """,
        
        "How many observations do I need?": """
        Minimum recommended sample size is 30, but more observations generally provide:
        - More stable results
        - Better ceiling line estimation
        - More reliable statistical tests
        """,
        
        "What about missing values?": """
        The tool can handle missing values through:
        - Automatic removal
        - Warning if > 20% missing
        - Documentation of handling in results
        """,
    }
    
    for question, answer in faqs.items():
        with st.expander(question):
            st.markdown(answer)

def render_troubleshooting() -> None:
    """Render troubleshooting section."""
    st.header("Troubleshooting")
    
    # Common Issues
    st.subheader("Common Issues")
    st.markdown("""
    1. **File Upload Problems**
        - Ensure file is in CSV format
        - Check file encoding (UTF-8 recommended)
        - Verify file isn't corrupted
    
    2. **Data Validation Errors**
        - Check minimum sample size (n ≥ 30)
        - Verify numeric columns
        - Review missing value percentage
    
    3. **Analysis Errors**
        - Verify variable selection
        - Check for constant values
        - Review preprocessing settings
    """)
    
    # Error Messages
    st.subheader("Error Messages")
    common_errors = {
        "Invalid file format": "Ensure file is CSV format",
        "Sample size too small": "Need at least 30 observations",
        "Non-numeric variables": "Convert variables to numeric",
        "Too many missing values": "Clean data or adjust threshold",
        "Constant variable detected": "Check variable variation",
    }
    
    for error, solution in common_errors.items():
        with st.expander(error):
            st.markdown(f"**Solution:** {solution}")
    
    # Data Problems
    st.subheader("Data Problems")
    st.markdown("""
    If you encounter data problems:
    
    1. **Check Data Quality**
        - Look for missing values
        - Identify outliers
        - Verify data types
    
    2. **Preprocessing Steps**
        - Try different scaling methods
        - Adjust outlier detection
        - Consider transformations
    
    3. **Analysis Settings**
        - Adjust method parameters
        - Try different methods
        - Review confidence levels
    """)
