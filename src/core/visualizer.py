"""Enhanced visualization functionality for NCA analysis."""
import numpy as np
import pandas as pd
from typing import Dict, Any
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class NCAVisualizer:
    """Handles visualization of NCA analysis results with enhanced plotting capabilities."""
    
    def __init__(self, results: Dict[str, Any]):
        self.results = results
        self.plot_config = {
            'height': 600,
            'template': 'plotly_white',
            'font_size': 12
        }
    
    def create_main_plot(self, x: np.ndarray, y: np.ndarray) -> go.Figure:
        """Create main NCA plot with ceiling line and confidence bands."""
        fig = go.Figure()
        
        # Add scatter plot of data points
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            name='Data Points',
            marker=dict(
                size=8,
                color='blue',
                opacity=0.6,
                line=dict(color='white', width=1)
            )
        ))
        
        # Handle ceiling line
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        ceiling_line = self.results['ceiling_line']
        
        if len(ceiling_line) != len(x):
            # Interpolate ceiling line to match data points
            interp_points = np.linspace(x.min(), x.max(), len(ceiling_line))
            ceiling_line = np.interp(x_sorted, interp_points, ceiling_line)
        
        # Add ceiling line
        fig.add_trace(go.Scatter(
            x=x_sorted,
            y=ceiling_line,
            mode='lines',
            name='Ceiling Line',
            line=dict(color='red', width=2)
        ))
        
        # Add confidence bands if available
        if 'bootstrap_results' in self.results:
            try:
                ci_upper = self.results['bootstrap_results']['statistics']['ceiling_ci_upper']
                ci_lower = self.results['bootstrap_results']['statistics']['ceiling_ci_lower']
                
                if len(ci_upper) != len(x):
                    # Interpolate confidence intervals
                    interp_points = np.linspace(x.min(), x.max(), len(ci_upper))
                    ci_upper = np.interp(x_sorted, interp_points, ci_upper)
                    ci_lower = np.interp(x_sorted, interp_points, ci_lower)
                
                fig.add_trace(go.Scatter(
                    x=x_sorted,
                    y=ci_upper,
                    mode='lines',
                    name='Upper CI',
                    line=dict(color='rgba(255,0,0,0.2)'),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=x_sorted,
                    y=ci_lower,
                    mode='lines',
                    name='Lower CI',
                    line=dict(color='rgba(255,0,0,0.2)'),
                    fill='tonexty',
                    showlegend=False
                ))
            except Exception as e:
                print(f"Warning: Could not add confidence bands: {str(e)}")
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='NCA Analysis Results',
                x=0.5,
                font=dict(size=20)
            ),
            xaxis_title='Independent Variable (X)',
            yaxis_title='Dependent Variable (Y)',
            height=self.plot_config['height'],
            template=self.plot_config['template'],
            hovermode='closest',
            showlegend=True
        )
        
        return fig
    
    def create_diagnostic_plots(self, x: np.ndarray, y: np.ndarray) -> go.Figure:
        """Create diagnostic plots for NCA analysis."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Residuals Distribution',
                'QQ Plot',
                'Residuals vs Predicted',
                'Effect Size Bootstrap Distribution'
            )
        )
        
        try:
            # Handle ceiling line interpolation for residuals
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            ceiling_line = self.results['ceiling_line']
            
            if len(ceiling_line) != len(x):
                interp_points = np.linspace(x.min(), x.max(), len(ceiling_line))
                ceiling_line = np.interp(x, interp_points, ceiling_line)
            
            # Calculate residuals
            residuals = y - ceiling_line
            
            # Residuals distribution
            fig.add_trace(
                go.Histogram(
                    x=residuals,
                    nbinsx=30,
                    name='Residuals',
                    marker_color='blue',
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            # QQ Plot
            sorted_residuals = np.sort(residuals)
            theoretical_quantiles = stats.norm.ppf(
                np.linspace(0.01, 0.99, len(residuals))
            )
            
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=sorted_residuals,
                    mode='markers',
                    name='QQ Plot',
                    marker=dict(color='blue')
                ),
                row=1, col=2
            )
            
            # Residuals vs Predicted
            fig.add_trace(
                go.Scatter(
                    x=ceiling_line,
                    y=residuals,
                    mode='markers',
                    name='Residuals vs Predicted',
                    marker=dict(color='blue')
                ),
                row=2, col=1
            )
            
            # Effect Size Bootstrap Distribution
            if 'bootstrap_results' in self.results and 'effect_sizes' in self.results['bootstrap_results']:
                fig.add_trace(
                    go.Histogram(
                        x=self.results['bootstrap_results']['effect_sizes'],
                        nbinsx=30,
                        name='Effect Size Distribution',
                        marker_color='blue',
                        opacity=0.7
                    ),
                    row=2, col=2
                )
        except Exception as e:
            print(f"Warning: Error creating diagnostic plots: {str(e)}")
            
        # Update layout
        fig.update_layout(
            height=800,
            template=self.plot_config['template'],
            showlegend=False
        )
        
        return fig
    
    def create_bottleneck_plot(self) -> go.Figure:
        """Create bottleneck analysis visualization."""
        try:
            bottleneck_df = self.results['bottleneck_table']
            
            fig = go.Figure()
            
            # Add main line
            fig.add_trace(go.Scatter(
                x=bottleneck_df['Y Level (%)'],
                y=bottleneck_df['Required X Level'],
                mode='lines+markers',
                name='Required X Level',
                line=dict(color='blue', width=2)
            ))
            
            # Highlight bottleneck regions
            bottleneck_regions = bottleneck_df[bottleneck_df['Is Bottleneck']]
            if len(bottleneck_regions) > 0:
                fig.add_trace(go.Scatter(
                    x=bottleneck_regions['Y Level (%)'],
                    y=bottleneck_regions['Required X Level'],
                    mode='markers',
                    name='Bottleneck Points',
                    marker=dict(
                        size=10,
                        color='red',
                        symbol='circle-dot'
                    )
                ))
            
            # Update layout
            fig.update_layout(
                title='Bottleneck Analysis',
                xaxis_title='Y Level (%)',
                yaxis_title='Required X Level',
                height=self.plot_config['height'],
                template=self.plot_config['template'],
                showlegend=True
            )
            
            return fig
        except Exception as e:
            print(f"Warning: Error creating bottleneck plot: {str(e)}")
            return go.Figure()
    
    def create_summary_cards(self) -> Dict[str, Any]:
        """Create summary statistics cards."""
        try:
            return {
                'effect_size': {
                    'value': f"{self.results['effect_size']:.3f}",
                    'ci': f"[{self.results['bootstrap_results']['statistics']['effect_size_ci'][0]:.3f}, "
                          f"{self.results['bootstrap_results']['statistics']['effect_size_ci'][1]:.3f}]"
                },
                'statistical_tests': {
                    'p_value': f"{self.results['statistical_tests']['permutation_p_value']:.3f}",
                    'normality': f"{self.results['statistical_tests']['normality_test_p']:.3f}"
                },
                'data_summary': {
                    'n_obs': str(self.results['data_summary']['n_observations']),
                    'correlation': f"{self.results['data_summary']['correlation']:.3f}"
                }
            }
        except Exception as e:
            print(f"Warning: Error creating summary cards: {str(e)}")
            return {}
