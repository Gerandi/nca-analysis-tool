"""Session state management utilities."""
import streamlit as st
from typing import Tuple, Optional, Any, Dict, List
import pandas as pd
from utils.constants import (
    ERROR_MESSAGES, 
    DEFAULT_SETTINGS, 
    ANALYSIS_METHODS,
    MIN_SAMPLE_SIZE,
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_CONFIDENCE_LEVEL
)

def initialize_session_state():
    """Initialize session state variables."""
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'x_var' not in st.session_state:
        st.session_state.x_var = None
    if 'y_var' not in st.session_state:
        st.session_state.y_var = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'analysis_settings' not in st.session_state:
        st.session_state.analysis_settings = {
            'method': 'CE-FDH',
            'bootstrap_iterations': DEFAULT_BOOTSTRAP_ITERATIONS,
            'confidence_level': DEFAULT_CONFIDENCE_LEVEL,
            'run_permutation_test': True,
            'compute_effect_size_ci': True
        }
    if 'preprocessing_settings' not in st.session_state:
        st.session_state.preprocessing_settings = {
            'outlier_method': 'none',
            'outlier_threshold': 3.0,
            'scaling_method': 'none',
            'transform_method': 'none'
        }
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'welcome'
    if 'data_prepared' not in st.session_state:
        st.session_state.data_prepared = False
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'export_settings' not in st.session_state:
        st.session_state.export_settings = {
            'last_format': None,
            'last_sections': None,
            'last_export_time': None
        }
    if 'page_completed' not in st.session_state:
        st.session_state.page_completed = set()

def get_analysis_results() -> Tuple[Optional[Dict[str, Any]], str]:
    """Get current analysis results with validation."""
    if not hasattr(st.session_state, 'analysis_results') or st.session_state.analysis_results is None:
        return None, ERROR_MESSAGES['NO_RESULTS']
    return st.session_state.analysis_results, ""

def get_current_data() -> Tuple[Optional[pd.DataFrame], str]:
    """Get current data with validation."""
    if not hasattr(st.session_state, 'data') or st.session_state.data is None:
        return None, ERROR_MESSAGES['NO_DATA']
    return st.session_state.data, ""

def update_analysis_settings(settings: Dict[str, Any]) -> None:
    """Update analysis settings with validation."""
    if not isinstance(settings, dict):
        raise ValueError(ERROR_MESSAGES['INVALID_SETTINGS'])
    
    # Initialize if not exists
    if 'analysis_settings' not in st.session_state:
        st.session_state.analysis_settings = {
            'method': 'CE-FDH',
            'bootstrap_iterations': DEFAULT_BOOTSTRAP_ITERATIONS,
            'confidence_level': DEFAULT_CONFIDENCE_LEVEL,
            'run_permutation_test': True,
            'compute_effect_size_ci': True
        }
    
    # Validate and update settings
    valid_keys = [
        'method', 
        'bootstrap_iterations', 
        'confidence_level',
        'run_permutation_test',
        'compute_effect_size_ci',
        'compute_diagnostics'
    ]
    
    for key, value in settings.items():
        if key in valid_keys:
            st.session_state.analysis_settings[key] = value
        else:
            st.warning(f"Ignoring invalid analysis setting: {key}")

def update_navigation_state(next_page: str) -> None:
    """Update navigation state and mark current page as completed."""
    # Store current page for completion tracking
    current_page = st.session_state.get('current_page')
    
    # Mark current page as completed
    if 'page_completed' not in st.session_state:
        st.session_state.page_completed = set()
    if current_page:
        st.session_state.page_completed.add(current_page)
    
    # Update current page
    st.session_state['current_page'] = next_page
    # Force a rerun to update the navigation
    st.rerun()

def can_proceed_to_analysis() -> bool:
    """Check if user can proceed to analysis page."""
    return (
        st.session_state.data is not None and 
        st.session_state.x_var is not None and 
        st.session_state.y_var is not None and 
        st.session_state.data_prepared
    )

def can_proceed_to_results() -> bool:
    """Check if user can proceed to results page."""
    return (
        can_proceed_to_analysis() and 
        st.session_state.analysis_results is not None and 
        st.session_state.analysis_complete
    )

def get_available_pages() -> List[str]:
    """Get list of available pages based on current state."""
    pages = ['welcome', 'data_prep', 'help']
    
    if can_proceed_to_analysis():
        pages.append('analysis')
        
    if can_proceed_to_results():
        pages.extend(['results', 'export'])
    
    return pages

def validate_analysis_settings() -> Tuple[bool, List[str]]:
    """Validate the current analysis settings."""
    errors = []
    
    # Check if analysis settings exist
    if not hasattr(st.session_state, 'analysis_settings'):
        errors.append(ERROR_MESSAGES['INVALID_SETTINGS'])
        return False, errors
    
    settings = st.session_state.analysis_settings
    
    # Validate method
    if 'method' not in settings or settings['method'] not in ANALYSIS_METHODS:
        errors.append(ERROR_MESSAGES['INVALID_METHOD'])
    
    # Validate bootstrap iterations
    if 'bootstrap_iterations' in settings:
        if not isinstance(settings['bootstrap_iterations'], int) or settings['bootstrap_iterations'] < 100:
            errors.append("Bootstrap iterations must be at least 100")
    
    # Validate confidence level
    if 'confidence_level' in settings:
        if not isinstance(settings['confidence_level'], (float, int)) or not 0 < settings['confidence_level'] < 1:
            errors.append("Confidence level must be between 0 and 1")
    
    return len(errors) == 0, errors

def update_preprocessing_settings(settings: Dict[str, Any]) -> None:
    """Update preprocessing settings with validation."""
    if not isinstance(settings, dict):
        raise ValueError(ERROR_MESSAGES['INVALID_SETTINGS'])
    
    # Initialize preprocessing settings if not present
    if 'preprocessing_settings' not in st.session_state:
        st.session_state.preprocessing_settings = {
            'outlier_method': 'none',
            'outlier_threshold': 3.0,
            'scaling_method': 'none',
            'transform_method': 'none'
        }
    
    # Validate and update settings
    valid_keys = [
        'outlier_method',
        'outlier_threshold',
        'scaling_method',
        'transform_method'
    ]
    
    for key, value in settings.items():
        if key in valid_keys:
            st.session_state.preprocessing_settings[key] = value
        else:
            st.warning(f"Ignoring invalid preprocessing setting: {key}")

def get_preprocessing_settings() -> Dict[str, Any]:
    """Get current preprocessing settings."""
    if 'preprocessing_settings' not in st.session_state:
        st.session_state.preprocessing_settings = {
            'outlier_method': 'none',
            'outlier_threshold': 3.0,
            'scaling_method': 'none',
            'transform_method': 'none'
        }
    return st.session_state.preprocessing_settings

def clear_analysis_results() -> None:
    """Clear analysis results and related settings."""
    st.session_state.analysis_results = None
    st.session_state.export_settings['last_export_time'] = None

def validate_session_state() -> Tuple[bool, str]:
    """Validate current session state."""
    if not hasattr(st.session_state, 'data') or st.session_state.data is None:
        return False, ERROR_MESSAGES['NO_DATA']
    if not hasattr(st.session_state, 'x_var') or st.session_state.x_var is None:
        return False, ERROR_MESSAGES['MISSING_VARIABLES']
    if not hasattr(st.session_state, 'y_var') or st.session_state.y_var is None:
        return False, ERROR_MESSAGES['MISSING_VARIABLES']
    return True, ""

def get_export_settings() -> Dict[str, Any]:
    """Get current export settings."""
    if not hasattr(st.session_state, 'export_settings'):
        st.session_state.export_settings = {
            'last_format': None,
            'last_sections': None,
            'last_export_time': None
        }
    return st.session_state.export_settings

def update_export_settings(format: str, sections: list) -> None:
    """Update export settings."""
    st.session_state.export_settings.update({
        'last_format': format,
        'last_sections': sections,
        'last_export_time': pd.Timestamp.now()
    })

def reset_settings() -> None:
    """Reset all settings to defaults."""
    st.session_state.preprocessing_settings = {
        'outlier_method': 'none',
        'outlier_threshold': 3.0,
        'scaling_method': 'none',
        'transform_method': 'none'
    }
    st.session_state.analysis_settings = {
        'method': 'CE-FDH',
        'bootstrap_iterations': DEFAULT_BOOTSTRAP_ITERATIONS,
        'confidence_level': DEFAULT_CONFIDENCE_LEVEL,
        'run_permutation_test': True,
        'compute_effect_size_ci': True
    }
    st.session_state.export_settings = {
        'last_format': None,
        'last_sections': None,
        'last_export_time': None
    }