"""Application constants and configurations."""

# Analysis Methods and Settings
ANALYSIS_METHODS = {
    "CE-FDH": {
        "name": "Ceiling Envelopment - Free Disposal Hull",
        "description": "Classic NCA method using step-wise ceiling line",
        "params": {}
    },
    "CR-FDH": {
        "name": "Ceiling Regression - Free Disposal Hull",
        "description": "Smooth ceiling line using local regression",
        "params": {
            "bandwidth": 0.1
        }
    },
    "Quantile": {
        "name": "Quantile Regression",
        "description": "Uses quantile regression for ceiling estimation",
        "params": {
            "quantile": 0.95
        }
    }
}

DEFAULT_BOOTSTRAP_ITERATIONS = 1000
DEFAULT_CONFIDENCE_LEVEL = 0.95

# File handling constants
SUPPORTED_FILE_TYPES = ["csv"]  # Add more file types if needed

# Data processing constants
MIN_SAMPLE_SIZE = 30
MAX_SAMPLE_SIZE = 1000000
MIN_ROWS = 10
SAMPLE_SIZE = 5
PREVIEW_ROWS = 10

# Preprocessing options
PREPROCESSING_OPTIONS = {
    "outlier_methods": [
        "none",
        "zscore",
        "iqr",
        "isolation_forest"
    ],
    "scaling_methods": [
        "none",
        "standard",
        "minmax",
        "robust"
    ],
    "transform_methods": [
        "none",
        "log",
        "sqrt",
        "box-cox"
    ],
    "thresholds": {
        "zscore": 3.0,
        "iqr": 1.5,
        "isolation_forest": 0.1
    }
}

# Custom CSS for styling
CUSTOM_CSS = """
<style>
    .stButton button {
        width: 100%;
    }
    
    .stDownloadButton button {
        background-color: #0078D7;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0.3rem;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    
    .stDownloadButton button:hover {
        background-color: #005A9E;
    }
    
    .error-message {
        color: #D32F2F;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.3rem;
        background-color: #FFEBEE;
    }
    
    .success-message {
        color: #388E3C;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.3rem;
        background-color: #E8F5E9;
    }
    
    .info-message {
        color: #1976D2;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.3rem;
        background-color: #E3F2FD;
    }
</style>
"""

# Report generation constants
REPORT_SECTIONS = {
    "EXECUTIVE_SUMMARY": "Executive Summary",
    "METHODOLOGY": "Methodology",
    "RESULTS": "Results",
    "STATISTICAL_DETAILS": "Statistical Details",
    "VISUALIZATIONS": "Visualizations",
    "RECOMMENDATIONS": "Recommendations",
    "APPENDIX": "Appendix"
}

# Export format configurations
EXPORT_FORMATS = {
    "CSV": {
        "mime_type": "text/csv",
        "extension": "csv",
        "default_options": {
            "delimiter": ",",
            "include_headers": True
        }
    },
    "EXCEL": {
        "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "extension": "xlsx",
        "default_options": {
            "include_charts": True,
            "auto_filter": True,
            "freeze_panes": True
        }
    },
    "JSON": {
        "mime_type": "application/json",
        "extension": "json",
        "default_options": {
            "indent": 2,
            "include_metadata": True
        }
    }
}

# Report templates
REPORT_TEMPLATES = {
    "PDF": {
        "page_size": "A4",
        "margins": {
            "top": 25,
            "right": 25,
            "bottom": 25,
            "left": 25
        },
        "header_spacing": 15,
        "footer_spacing": 15
    },
    "LATEX": {
        "document_classes": ["article", "report", "paper"],
        "bibliography_styles": ["plain", "apa", "ieee"],
        "default_packages": [
            "geometry",
            "graphicx",
            "booktabs",
            "hyperref",
            "float"
        ]
    }
}

# Error messages
ERROR_MESSAGES = {
    "NO_DATA": "No data available. Please upload data first.",
    "NO_RESULTS": "No analysis results available. Please run the analysis first.",
    "INVALID_FORMAT": "Invalid export format selected.",
    "GENERATION_FAILED": "Failed to generate report. Please check your settings.",
    "EXPORT_FAILED": "Failed to export data. Please try again.",
    "MISSING_VARIABLES": "Required variables are missing.",
    "INVALID_SETTINGS": "Invalid settings provided.",
    "SHAPE_MISMATCH": "Data shape mismatch. Please rerun the analysis.",
    "INITIALIZATION_ERROR": "Error initializing report generator.",
    "EXPORT_ERROR": "Error during export process.",
    "FILE_TOO_LARGE": "File size exceeds maximum limit.",
    "INVALID_FILE_TYPE": "Unsupported file type. Please upload a CSV or Excel file.",
    "PARSING_ERROR": "Error parsing file. Please check the file format.",
    "MISSING_COLUMNS": "Required columns are missing from the dataset.",
    "SAMPLE_SIZE_ERROR": f"Sample size must be at least {MIN_SAMPLE_SIZE}",
    "INVALID_METHOD": "Invalid analysis method selected.",
    "INVALID_PARAMS": "Invalid method parameters provided."
}

# Success messages
SUCCESS_MESSAGES = {
    "REPORT_GENERATED": "Report generated successfully!",
    "EXPORT_COMPLETED": "Data exported successfully!",
    "SETTINGS_SAVED": "Settings saved successfully!",
    "FILE_UPLOADED": "File uploaded successfully!",
    "ANALYSIS_COMPLETE": "Analysis completed successfully!"
}

# File patterns
FILE_PATTERNS = {
    "REPORT": "nca_report_{timestamp}.{extension}",
    "EXPORT": "nca_export_{timestamp}.{extension}",
    "TIMESTAMP_FORMAT": "%Y%m%d_%H%M%S"
}

# Default settings
DEFAULT_SETTINGS = {
    "REPORT": {
        "pdf_toc": True,
        "pdf_pages": True,
        "pdf_citations": True,
        "pdf_refs": True,
        "excel_raw": True,
        "excel_filter": True,
        "excel_charts": True,
        "latex_class": "article",
        "latex_abstract": True,
        "latex_bib": "plain"
    },
    "EXPORT": {
        "csv_delimiter": ",",
        "csv_headers": True,
        "json_pretty": True,
        "json_meta": True,
        "excel_sheets": True,
        "excel_format": True
    }
}

# Analysis settings
ANALYSIS_SETTINGS = {
    "MIN_SAMPLE_SIZE": MIN_SAMPLE_SIZE,
    "DEFAULT_CONFIDENCE": DEFAULT_CONFIDENCE_LEVEL,
    "BOOTSTRAP_ITERATIONS": DEFAULT_BOOTSTRAP_ITERATIONS,
    "MAX_OUTLIER_PERCENTAGE": 10
}

# Variable types
VARIABLE_TYPES = {
    "NUMERIC": ["int64", "float64"],
    "CATEGORICAL": ["object", "category", "bool"],
    "DATETIME": ["datetime64"],
}

# Help messages
HELP_MESSAGES = {
    "data_upload": "Upload your data file in CSV or Excel format.",
    "variable_selection": "Select the independent (X) and dependent (Y) variables for analysis.",
    "analysis_settings": "Configure analysis parameters and settings.",
    "report_options": "Customize your report with various sections and formats.",
    "export_options": "Choose how to export your analysis results and data."
}

# UI configurations
UI_CONFIGS = {
    "SIDEBAR_WIDTH": 300,
    "PLOT_HEIGHT": 500,
    "PLOT_WIDTH": 800,
    "MAX_DISPLAY_ROWS": 1000,
    "DECIMAL_PLACES": 3
}
