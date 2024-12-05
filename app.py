"""Application entry point."""
import os
import sys
from pathlib import Path

# Add src directory to Python path
src_path = str(Path(__file__).parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import and run the main application
import streamlit as st
from src.app import main

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please try refreshing the page or contact support if the issue persists.")
