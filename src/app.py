"""Main application entry point."""
import streamlit as st
from typing import Dict
from utils.constants import CUSTOM_CSS
from utils.session_state import initialize_session_state, get_available_pages
from ui.pages import welcome, data_prep, analysis, help

# Configure page settings
st.set_page_config(
    page_title="NCA Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page mappings
PAGES: Dict[str, callable] = {
    'welcome': welcome.render,
    'data_prep': data_prep.render,
    'analysis': analysis.render,
    'help': help.render
}

# Page titles for display
PAGE_TITLES = {
    'welcome': "Welcome",
    'data_prep': "Data Preparation",
    'analysis': "Analysis & Results",
    'help': "Help & Documentation"
}

def render_navigation():
    """Render the navigation sidebar."""
    st.sidebar.title("Navigation")
    
    # Get current page from session state
    current_page = st.session_state.get('current_page', 'welcome')
    
    # Select current page
    selected_page = st.sidebar.radio(
        "Go to",
        list(PAGES.keys()),
        format_func=lambda x: PAGE_TITLES.get(x, x.capitalize()),
        index=list(PAGES.keys()).index(current_page) if current_page in list(PAGES.keys()) else 0
    )
    
    # Update current page in session state if changed
    if selected_page != current_page:
        st.session_state.current_page = selected_page
        st.rerun()
    
    return selected_page

def render_progress_indicator():
    """Render progress indicator in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Progress")
    
    # Define progress steps
    steps = {
        'Data Upload': st.session_state.get('data') is not None,
        'Variable Selection': st.session_state.get('x_var') is not None and st.session_state.get('y_var') is not None,
        'Preprocessing': st.session_state.get('data_prepared', False),
        'Analysis': st.session_state.get('analysis_results') is not None,
        'Results': st.session_state.get('analysis_complete', False)
    }
    
    # Display progress
    for step, completed in steps.items():
        icon = "✅" if completed else "⭕"
        st.sidebar.markdown(f"{icon} {step}")

def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()
    
    # Apply custom styling
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Render navigation and get current page
    current_page = render_navigation()
    
    # Render progress indicator
    render_progress_indicator()
    
    # Render selected page
    PAGES[current_page]()
    
    # Add footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "NCA Analysis Tool  \n"
        "Version 1.0.0  \n"
        "© 2024"
    )

if __name__ == "__main__":
    main()