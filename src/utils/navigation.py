"""Navigation utility functions."""
import streamlit as st
from typing import Dict, List

# Page titles for display
PAGE_TITLES = {
    'welcome': "Welcome",
    'data_prep': "Data Preparation",
    'analysis': "Analysis Configuration",
    'results': "Analysis Results",
    'export': "Export Results",
    'help': "Help & Documentation"
}

def get_navigation_status() -> Dict[str, bool]:
    """Get the status of each navigation step."""
    return {
        'Data Upload': st.session_state.data is not None,
        'Variable Selection': (
            st.session_state.x_var is not None and 
            st.session_state.y_var is not None
        ),
        'Preprocessing': st.session_state.get('data_prepared', False),
        'Analysis': st.session_state.get('analysis_results') is not None,
        'Results': st.session_state.get('analysis_complete', False)
    }

def render_navigation_progress():
    """Render the progress indicator in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis Progress")
    
    # Get current status
    status = get_navigation_status()
    
    # Display progress
    for step, completed in status.items():
        icon = "✅" if completed else "⭕"
        st.sidebar.markdown(f"{icon} {step}")
    
def can_navigate_to(page: str) -> bool:
    """Check if navigation to a specific page is allowed."""
    status = get_navigation_status()
    
    if page == 'welcome':
        return True
    elif page == 'data_prep':
        return True
    elif page == 'analysis':
        return status['Data Upload'] and status['Variable Selection'] and status['Preprocessing']
    elif page == 'results':
        return status['Analysis']
    elif page == 'export':
        return status['Results']
    elif page == 'help':
        return True
    return False