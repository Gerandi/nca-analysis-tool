"""Report generation functionality."""
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Optional, Union
from io import BytesIO
from datetime import datetime
from fpdf import FPDF
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class NCAPDFReport(FPDF):
    """Enhanced PDF report generator with better formatting and layout."""

    def __init__(self):
        super().__init__()
        self.WIDTH = 210
        self.HEIGHT = 297
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(left=10, top=10, right=10)
        self._current_section = ""

    def header(self):
        """Add custom header to each page."""
        if self.page_no() > 1:  # Skip header on first page
            self.set_font('Arial', 'I', 8)
            self.set_text_color(128)
            self.cell(0, 10, f'NCA Analysis Report - {self._current_section}', 0, 0, 'L')
            self.cell(0, 10, f'Page {self.page_no()}', 0, 1, 'R')
            self.ln(5)

    def footer(self):
        """Add custom footer to each page."""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 0, 'C')

    def add_title_page(self, title: str, subtitle: Optional[str] = None):
        """Create an attractive title page."""
        self.add_page()
        self._current_section = "Title"

        self.ln(60)  # Add spacing
        self.set_font('Arial', 'B', 24)
        self.set_text_color(44, 62, 80)
        self.cell(0, 15, title, ln=True, align='C')

        if subtitle:
            self.ln(10)
            self.set_font('Arial', 'I', 14)
            self.set_text_color(52, 73, 94)
            self.cell(0, 10, subtitle, ln=True, align='C')

        self.ln(20)
        self.set_font('Arial', '', 12)
        self.set_text_color(0)
        self.cell(0, 10, f'Generated on: {datetime.now().strftime("%B %d, %Y at %H:%M")}', 
                   ln=True, align='C')

    def add_chapter(self, title: str):
        """Add formatted chapter title."""
        self.add_page()
        self._current_section = title
        self.set_font('Arial', 'B', 14)
        self.set_text_color(44, 62, 80)
        self.set_fill_color(236, 240, 241)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(5)

    def add_section(self, title: str):
        """Add formatted section title."""
        self.set_font('Arial', 'B', 12)
        self.set_text_color(52, 73, 94)
        self.cell(0, 8, title, 0, 1, 'L')
        self.ln(2)

    def add_text(self, text: str):
        """Add formatted body text."""
        self.set_font('Arial', '', 10)
        self.set_text_color(0)
        self.multi_cell(0, 5, text)
        self.ln(3)

    def add_metric(self, label: str, value: Union[str, float, int], interpretation: Optional[str] = None):
        """Add a metric with optional interpretation."""
        self.set_font('Arial', 'B', 11)
        self.cell(60, 7, label, 0, 0)

        self.set_font('Arial', '', 11)
        if isinstance(value, (float, np.floating)):
            self.cell(30, 7, f'{value:.3f}', 0, 1)
        else:
            self.cell(30, 7, str(value), 0, 1)

        if interpretation:
            self.set_font('Arial', 'I', 9)
            self.set_text_color(96, 96, 96)
            self.cell(0, 5, f'→ {interpretation}', 0, 1, 'L')
            self.set_text_color(0)

        self.ln(2)

    def add_table(self, headers: List[str], data: List[List[Any]], widths: Optional[List[int]] = None):
        """Add a formatted table."""
        if widths is None:
            width = self.get_string_width('M')
            widths = [30 * width / len(headers)] * len(headers)

        self.set_font('Arial', 'B', 10)
        self.set_fill_color(236, 240, 241)
        for i, header in enumerate(headers):
            self.cell(widths[i], 7, str(header), 1, 0, 'C', 1)
        self.ln()

        self.set_font('Arial', '', 9)
        for row in data:
            for i, cell in enumerate(row):
                self.cell(widths[i], 6, str(cell), 1, 0, 'C')
            self.ln()

        self.ln(3)

class ReportGenerator:
    """Enhanced report generation with comprehensive analysis."""

    def __init__(
        self,
        results: Dict[str, Any],
        data: pd.DataFrame,
        x_var: str,
        y_var: str,
        settings: Dict[str, Any]
    ):
        self.results = results
        self.data = data
        self.x_var = x_var
        self.y_var = y_var
        self.settings = settings
        self.timestamp = pd.Timestamp.now()

        # Validate inputs and calculate additional metrics
        self._validate_inputs()
        self._calculate_additional_metrics()

    def _validate_inputs(self):
        """Validate input data and results."""
        required_results = [
            'effect_size', 'ceiling_line', 'bottleneck_table',
            'statistical_tests', 'effect_stats', 'data_summary'
        ]

        missing = [key for key in required_results if key not in self.results]
        if missing:
            raise ValueError(f"Missing required results: {', '.join(missing)}")

        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")

        if self.x_var not in self.data.columns or self.y_var not in self.data.columns:
            raise ValueError("Specified variables not found in data")

    def _calculate_additional_metrics(self):
        """Calculate additional analysis metrics."""
        self.additional_metrics = {
            'effect_size_interpretation': self._interpret_effect_size(),
            'significance_interpretation': self._interpret_significance(),
            'model_fit': self._calculate_model_fit(),
            'bottleneck_summary': self._summarize_bottlenecks(),
            'data_quality': self._assess_data_quality()
        }

    def _interpret_effect_size(self) -> Dict[str, Any]:
        """Interpret the effect size results."""
        effect_size = self.results['effect_size']

        if effect_size >= 0.5:
            level = "Strong"
            description = "Strong necessary condition relationship"
        elif effect_size >= 0.3:
            level = "Moderate"
            description = "Moderate necessary condition relationship"
        elif effect_size >= 0.1:
            level = "Weak"
            description = "Weak necessary condition relationship"
        else:
            level = "Negligible"
            description = "Negligible necessary condition relationship"

        return {
            'level': level,
            'description': description,
            'value': effect_size,
            'confidence_interval': self.results['bootstrap_results']['statistics']['effect_size_ci']
        }

    def _interpret_significance(self) -> Dict[str, Any]:
        """Interpret statistical significance results."""
        p_value = self.results['statistical_tests']['permutation_p_value']

        return {
            'significant': p_value < 0.05,
            'p_value': p_value,
            'interpretation': "Statistically significant" if p_value < 0.05 else "Not statistically significant",
            'confidence_level': (1 - p_value) * 100
        }

    def _calculate_model_fit(self) -> Dict[str, Any]:
        """Calculate and interpret model fit metrics."""
        residuals = self.data[self.y_var].values - self.results['ceiling_line']
        r_squared = 1 - np.var(residuals) / np.var(self.data[self.y_var])

        return {
            'r_squared': r_squared,
            'residual_std': np.std(residuals),
            'mean_absolute_error': np.mean(np.abs(residuals))
        }

    def _summarize_bottlenecks(self) -> Dict[str, Any]:
        """Summarize bottleneck analysis results."""
        bottleneck_df = self.results['bottleneck_table']
        critical_levels = bottleneck_df[bottleneck_df['Is Bottleneck']]['Y Level (%)'].values

        return {
            'critical_levels': critical_levels.tolist(),
            'n_bottlenecks': len(critical_levels),
            'max_constraint': float(bottleneck_df['Required X Level'].max()),
            'min_constraint': float(bottleneck_df['Required X Level'].min()),
            'bottleneck_ranges': self._identify_bottleneck_ranges(bottleneck_df)
        }

    def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess the quality of input data."""
        return {
            'sample_size': len(self.data),
            'missing_values': self.data[[self.x_var, self.y_var]].isnull().sum().to_dict(),
            'outliers': {
                'x': self._detect_outliers(self.data[self.x_var]),
                'y': self._detect_outliers(self.data[self.y_var])
            },
            'variable_ranges': {
                'x': {
                    'min': float(self.data[self.x_var].min()),
                    'max': float(self.data[self.x_var].max()),
                    'range': float(self.data[self.x_var].max() - self.data[self.x_var].min())
                },
                'y': {
                    'min': float(self.data[self.y_var].min()),
                    'max': float(self.data[self.y_var].max()),
                    'range': float(self.data[self.y_var].max() - self.data[self.y_var].min())
                }
            }
        }

def detect_outliers(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Detect outliers with proper NaN handling using multiple methods."""
    results = {}
    
    try:
        # Create a mask for non-NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        
        if len(x_clean) == 0:
            return {
                'outliers_x': [],
                'outliers_y': [],
                'outlier_mask': np.zeros_like(x, dtype=bool),
                'method': 'none',
                'message': 'No valid data points after NaN removal'
            }
        
        # Combine features for isolation forest
        X_combined = np.column_stack([x_clean, y_clean])
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination='auto',
            random_state=42,
            n_jobs=-1
        )
        
        outlier_labels = iso_forest.fit_predict(X_scaled)
        outlier_mask_clean = outlier_labels == -1
        
        # Create full-size mask matching original data
        full_outlier_mask = np.zeros_like(x, dtype=bool)
        full_outlier_mask[valid_mask] = outlier_mask_clean
        
        # Get indices of outliers
        outlier_indices = np.where(full_outlier_mask)[0]
        
        results = {
            'outliers_x': x[outlier_indices].tolist(),
            'outliers_y': y[outlier_indices].tolist(),
            'outlier_mask': full_outlier_mask,
            'method': 'IsolationForest',
            'n_outliers': len(outlier_indices),
            'outlier_indices': outlier_indices.tolist(),
            'percentage': (len(outlier_indices) / len(x)) * 100
        }
        
        # Add backup IQR detection
        iqr_x = np.percentile(x_clean, 75) - np.percentile(x_clean, 25)
        iqr_y = np.percentile(y_clean, 75) - np.percentile(y_clean, 25)
        
        q1_x = np.percentile(x_clean, 25)
        q3_x = np.percentile(x_clean, 75)
        q1_y = np.percentile(y_clean, 25)
        q3_y = np.percentile(y_clean, 75)
        
        iqr_outliers_x = x_clean[(x_clean < (q1_x - 1.5 * iqr_x)) | (x_clean > (q3_x + 1.5 * iqr_x))]
        iqr_outliers_y = y_clean[(y_clean < (q1_y - 1.5 * iqr_y)) | (y_clean > (q3_y + 1.5 * iqr_y))]
        
        results['iqr_outliers'] = {
            'x': iqr_outliers_x.tolist(),
            'y': iqr_outliers_y.tolist(),
            'n_outliers_x': len(iqr_outliers_x),
            'n_outliers_y': len(iqr_outliers_y)
        }
        
    except Exception as e:
        # Fallback to simple IQR method if Isolation Forest fails
        results = self._fallback_outlier_detection(x, y)
        results['error'] = str(e)
        results['method'] = 'IQR_fallback'
    
    return results

def _fallback_outlier_detection(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Fallback outlier detection using IQR method when Isolation Forest fails."""
    # Remove NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    
    if len(x_clean) == 0:
        return {
            'outliers_x': [],
            'outliers_y': [],
            'outlier_mask': np.zeros_like(x, dtype=bool),
            'method': 'none',
            'message': 'No valid data points after NaN removal'
        }
    
    # Calculate IQR for both variables
    q1_x, q3_x = np.percentile(x_clean, [25, 75])
    q1_y, q3_y = np.percentile(y_clean, [25, 75])
    iqr_x = q3_x - q1_x
    iqr_y = q3_y - q1_y
    
    # Define bounds
    x_lower = q1_x - 1.5 * iqr_x
    x_upper = q3_x + 1.5 * iqr_x
    y_lower = q1_y - 1.5 * iqr_y
    y_upper = q3_y + 1.5 * iqr_y
    
    # Create outlier mask
    outlier_mask_clean = ((x_clean < x_lower) | (x_clean > x_upper) |
                         (y_clean < y_lower) | (y_clean > y_upper))
    
    # Create full-size mask matching original data
    full_outlier_mask = np.zeros_like(x, dtype=bool)
    full_outlier_mask[valid_mask] = outlier_mask_clean
    
    # Get outlier indices
    outlier_indices = np.where(full_outlier_mask)[0]
    
    return {
        'outliers_x': x[outlier_indices].tolist(),
        'outliers_y': y[outlier_indices].tolist(),
        'outlier_mask': full_outlier_mask,
        'method': 'IQR',
        'n_outliers': len(outlier_indices),
        'outlier_indices': outlier_indices.tolist(),
        'percentage': (len(outlier_indices) / len(x)) * 100,
        'bounds': {
            'x': {'lower': float(x_lower), 'upper': float(x_upper)},
            'y': {'lower': float(y_lower), 'upper': float(y_upper)}
        }
    }

    def _identify_bottleneck_ranges(self, bottleneck_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify continuous ranges of bottlenecks."""
        ranges = []
        current_range = None

        for _, row in bottleneck_df.iterrows():
            if row['Is Bottleneck']:
                if current_range is None:
                    current_range = {
                        'start_y': row['Y Level (%)'],
                        'start_x': row['Required X Level']
                    }
            elif current_range is not None:
                current_range['end_y'] = row['Y Level (%)']
                current_range['end_x'] = row['Required X Level']
                ranges.append(current_range)
                current_range = None

        return ranges

    def _generate_executive_summary(self) -> str:
        """Generate executive summary text."""
        effect_size = self.additional_metrics['effect_size_interpretation']
        significance = self.additional_metrics['significance_interpretation']
        bottleneck = self.additional_metrics['bottleneck_summary']

        summary = (
            f"Analysis of the relationship between {self.x_var} and {self.y_var} reveals a "
            f"{effect_size['level'].lower()} necessary condition effect (d = {effect_size['value']:.3f}). "
            f"This relationship is {significance['interpretation'].lower()}. "
            f"\n\nThe analysis identified {bottleneck['n_bottlenecks']} critical bottleneck levels "
            f"where {self.x_var} constrains {self.y_var}. "
            f"The required level of {self.x_var} ranges from {bottleneck['min_constraint']:.2f} "
            f"to {bottleneck['max_constraint']:.2f}."
        )

        return summary

    def _add_analysis_results(self, report: NCAPDFReport) -> None:
        """Add analysis results section to the report."""
        report.add_section("Effect Size Analysis")
        effect_size = self.additional_metrics['effect_size_interpretation']
        report.add_metric(
            "Effect Size (d)", 
            effect_size['value'],
            effect_size['description']
        )
        report.add_metric(
            "95% CI",
            f"[{effect_size['confidence_interval'][0]:.3f}, {effect_size['confidence_interval'][1]:.3f}]"
        )

        report.add_section("Statistical Tests")
        significance = self.additional_metrics['significance_interpretation']
        report.add_metric(
            "P-value",
            significance['p_value'],
            significance['interpretation']
        )

        report.add_section("Model Fit")
        fit = self.additional_metrics['model_fit']
        report.add_metric("R-squared", fit['r_squared'])
        report.add_metric("Residual Std", fit['residual_std'])
        report.add_metric("Mean Absolute Error", fit['mean_absolute_error'])

    def _add_statistical_details(self, report: NCAPDFReport) -> None:
        """Add detailed statistical analysis to the report."""
        report.add_section("Data Quality Assessment")
        quality = self.additional_metrics['data_quality']

        report.add_metric("Sample Size", quality['sample_size'])
        report.add_metric(
            "Outliers",
            f"X: {quality['outliers']['x']['count']} ({quality['outliers']['x']['percentage']:.1f}%), "
            f"Y: {quality['outliers']['y']['count']} ({quality['outliers']['y']['percentage']:.1f}%)"
        )

        report.add_section("Variable Ranges")
        x_range = quality['variable_ranges']['x']
        y_range = quality['variable_ranges']['y']
        report.add_metric(f"{self.x_var} Range", f"{x_range['min']:.2f} - {x_range['max']:.2f}")
        report.add_metric(f"{self.y_var} Range", f"{y_range['min']:.2f} - {y_range['max']:.2f}")

    def generate_pdf_report(self) -> bytes:
        """Generate comprehensive PDF report."""
        report = NCAPDFReport()

        # Title page
        report.add_title_page(
            "Necessary Condition Analysis Report",
            f"Analysis of {self.x_var} → {self.y_var}"
        )

        # Executive Summary
        report.add_chapter("Executive Summary")
        report.add_text(self._generate_executive_summary())

        # Main Analysis Results
        report.add_chapter("Analysis Results")
        self._add_analysis_results(report)

        # Statistical Details
        report.add_chapter("Statistical Details")
        self._add_statistical_details(report)

        # Bottleneck Analysis
        report.add_chapter("Bottleneck Analysis")
        bottleneck = self.additional_metrics['bottleneck_summary']
        report.add_metric("Number of Bottlenecks", bottleneck['n_bottlenecks'])
        if bottleneck['n_bottlenecks'] > 0:
            report.add_table(
                ["Y Level (%)", "Required X Level"],
                [[level, self.results['bottleneck_table'].loc[i, 'Required X Level']] 
                 for i, level in enumerate(bottleneck['critical_levels'])]
            )

        return report.output(dest='S').encode('latin1')

    def generate_latex_report(self) -> str:
        """Generate LaTeX formatted report."""
        template = (
            "\\documentclass{article}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{graphicx}\n"
            "\\usepackage{float}\n"
            "\\begin{document}\n"
            "\\title{Necessary Condition Analysis Report}\n"
            f"\\author{{{self.x_var} → {self.y_var}}}\n"
            "\\date{\\today}\n"
            "\\maketitle\n\n"
        )

        # Add sections
        template += (
            "\\section{Executive Summary}\n"
            f"{self._generate_executive_summary()}\n\n"
            "\\section{Analysis Results}\n"
            # Additional sections can be added here
            "\\end{document}"
        )

        return template
    def generate_excel_report(self) -> bytes:
        """Generate Excel report with multiple sheets."""
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Summary sheet
            summary_data = {
                'Metric': ['Effect Size', 'P-value', 'R-squared'],
                'Value': [
                    self.results['effect_size'],
                    self.results['statistical_tests']['permutation_p_value'],
                    self.additional_metrics['model_fit']['r_squared']
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

            # Raw data
            self.data.to_excel(writer, sheet_name='Raw Data', index=False)

            # Bottleneck analysis
            self.results['bottleneck_table'].to_excel(
                writer, sheet_name='Bottleneck Analysis', index=False
            )

        return output.getvalue()

    def export_to_json(self, content_selection: List[str]) -> bytes:
        """Export selected content to JSON format."""
        export_data = {
            'metadata': {
                'timestamp': str(self.timestamp),
                'variables': {
                    'x': self.x_var,
                    'y': self.y_var
                }
            }
        }

        if 'Analysis Results' in content_selection:
            export_data['results'] = {
                'effect_size': self.results['effect_size'],
                'significance': self.additional_metrics['significance_interpretation'],
                'model_fit': self.additional_metrics['model_fit']
            }

        if 'Raw Data' in content_selection:
            export_data['raw_data'] = self.data.to_dict(orient='records')

        if 'Statistical Tests' in content_selection:
            export_data['statistical_tests'] = self.results['statistical_tests']

        if 'Bottleneck Analysis' in content_selection:
            export_data['bottleneck_analysis'] = {
                'summary': self.additional_metrics['bottleneck_summary'],
                'details': self.results['bottleneck_table'].to_dict(orient='records')
            }

        return json.dumps(export_data, indent=2).encode('utf-8')

    def export_to_csv(self, content_selection: List[str]) -> bytes:
        """Export selected content to CSV format."""
        output = BytesIO()

        if 'Raw Data' in content_selection:
            self.data.to_csv(output, index=False)
        elif 'Bottleneck Analysis' in content_selection:
            self.results['bottleneck_table'].to_csv(output, index=False)
        else:
            # Create summary DataFrame
            summary_data = pd.DataFrame({
                'Metric': ['Effect Size', 'P-value', 'R-squared'],
                'Value': [
                    self.results['effect_size'],
                    self.results['statistical_tests']['permutation_p_value'],
                    self.additional_metrics['model_fit']['r_squared']
                ]
            })
            summary_data.to_csv(output, index=False)

        return output.getvalue()

    def export_to_excel(self, content_selection: List[str]) -> bytes:
        """Export selected content to Excel format."""
        output = BytesIO()

        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            if 'Raw Data' in content_selection:
                self.data.to_excel(writer, sheet_name='Raw Data', index=False)

            if 'Analysis Results' in content_selection:
                summary_data = pd.DataFrame({
                    'Metric': ['Effect Size', 'P-value', 'R-squared'],
                    'Value': [
                        self.results['effect_size'],
                        self.results['statistical_tests']['permutation_p_value'],
                        self.additional_metrics['model_fit']['r_squared']
                    ]
                })
                summary_data.to_excel(writer, sheet_name='Summary', index=False)

            if 'Bottleneck Analysis' in content_selection:
                self.results['bottleneck_table'].to_excel(
                    writer, sheet_name='Bottleneck Analysis', index=False
                )

        return output.getvalue()
