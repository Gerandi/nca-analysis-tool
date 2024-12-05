"""Core NCA analysis functionality."""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional
from joblib import Parallel, delayed
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import QuantileRegressor


class NCAAnalyzer:
    """Core NCA analysis engine with enhanced functionality and error handling."""

    def __init__(self, method: str = 'CE-FDH'):
        self.method = method
        self.results: Optional[Dict[str, Any]] = None
        self.diagnostics: Dict[str, Any] = {}
        self.validation_results: Dict[str, Any] = {}

    def validate_inputs(self, x: np.ndarray, y: np.ndarray) -> Tuple[bool, List[str]]:
        """Validate input data for NCA analysis."""
        errors = []

        if len(x) < 30:  # MIN_SAMPLE_SIZE
            errors.append(f"Sample size ({len(x)}) is below minimum required (30)")

        if len(x) != len(y):
            errors.append("X and Y arrays must have equal length")

        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            errors.append("Data contains non-finite values")

        if np.all(x == x[0]) or np.all(y == y[0]):
            errors.append("Data contains constant values")

        return len(errors) == 0, errors

    def ce_fdh(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Calculate Ceiling Line using CE-FDH method."""
        stats = {'method': 'CE-FDH'}

        try:
            sorted_indices = np.argsort(x)
            x_sorted = x[sorted_indices]
            y_sorted = y[sorted_indices]

            ceiling_y = np.zeros_like(y_sorted)
            ceiling_y[0] = y_sorted[0]

            for i in range(1, len(x_sorted)):
                ceiling_y[i] = max(ceiling_y[i - 1], y_sorted[i])

            # Reorder back to original indices
            ceiling_y = ceiling_y[np.argsort(sorted_indices)]

            # Calculate additional statistics
            stats.update({
                'ceiling_points': np.sum(y == ceiling_y),
                'ceiling_percentage': (np.sum(y == ceiling_y) / len(y)) * 100,
                'average_gap': np.mean(ceiling_y - y)
            })

            return ceiling_y, stats

        except Exception as e:
            raise ValueError(f"Error in CE-FDH computation: {str(e)}")

    def cr_fdh(self, x: np.ndarray, y: np.ndarray, bandwidth: float = 0.1) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Calculate Ceiling Line using CR-FDH method."""
        stats = {
            'method': 'CR-FDH',
            'bandwidth': bandwidth
        }

        try:
            x_range = np.max(x) - np.min(x)
            y_ceiling = np.zeros_like(y)
            window_counts = np.zeros_like(y)

            for i in range(len(x)):
                window = np.abs(x - x[i]) <= (bandwidth * x_range)
                if np.sum(window) > 0:
                    y_ceiling[i] = np.max(y[window])
                    window_counts[i] = np.sum(window)
                else:
                    y_ceiling[i] = y[i]
                    window_counts[i] = 1

            stats.update({
                'average_window_size': float(np.mean(window_counts)),
                'min_window_size': int(np.min(window_counts)),
                'max_window_size': int(np.max(window_counts))
            })

            return y_ceiling, stats

        except Exception as e:
            raise ValueError(f"Error in CR-FDH computation: {str(e)}")

    def quantile_regression(self, x: np.ndarray, y: np.ndarray,
                            quantile: float = 0.95) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Calculate ceiling line using quantile regression."""
        stats = {
            'method': 'Quantile Regression',
            'quantile': quantile
        }

        try:
            qr = QuantileRegressor(quantile=quantile, solver="highs", alpha=0.0)
            X = x.reshape(-1, 1)
            qr.fit(X, y)
            ceiling_y = qr.predict(X)

            stats.update({
                'intercept': float(qr.intercept_),
                'coefficient': float(qr.coef_[0]),
                'solver_status': str(qr.n_iter_)
            })

            return ceiling_y, stats

        except Exception as e:
            raise ValueError(f"Error in quantile regression computation: {str(e)}")
    def calculate_effect_size(self, x: np.ndarray, y: np.ndarray, ceiling_y: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Calculate NCA effect size with comprehensive statistics."""
        stats = {}

        try:
            y_range = np.max(y) - np.min(y)
            if y_range == 0:
                return 0.0, {'error': 'Zero range in Y variable'}

            ceiling_effect = np.sum(ceiling_y - np.min(y)) / len(y)
            effect_size = ceiling_effect / y_range

            stats.update({
                'raw_effect': float(ceiling_effect),
                'y_range': float(y_range),
                'effect_size': float(effect_size),
                'ceiling_mean': float(np.mean(ceiling_y)),
                'ceiling_std': float(np.std(ceiling_y)),
                'points_at_ceiling': int(np.sum(np.isclose(y, ceiling_y, rtol=1e-5))),
                'points_below_ceiling': int(np.sum(y < ceiling_y))
            })

            return effect_size, stats

        except Exception as e:
            raise ValueError(f"Error in effect size calculation: {str(e)}")

    def bootstrap_analysis(self, x: np.ndarray, y: np.ndarray, n_iterations: int = 1000) -> Dict[str, Any]:
        """Perform bootstrap analysis with enhanced statistics."""
        bootstrap_results = {
            'effect_sizes': np.zeros(n_iterations),
            'ceiling_lines': np.zeros((n_iterations, len(x))),
            'statistics': {}
        }

        try:
            def single_bootstrap(i: int) -> Tuple[float, np.ndarray]:
                indices = np.random.choice(len(x), len(x), replace=True)
                x_boot, y_boot = x[indices], y[indices]

                if self.method == 'CE-FDH':
                    ceiling, _ = self.ce_fdh(x_boot, y_boot)
                elif self.method == 'CR-FDH':
                    ceiling, _ = self.cr_fdh(x_boot, y_boot)
                else:
                    ceiling, _ = self.quantile_regression(x_boot, y_boot)

                effect_size, _ = self.calculate_effect_size(x_boot, y_boot, ceiling)
                return effect_size, ceiling

            # Parallel bootstrap computation
            results = Parallel(n_jobs=-1)(
                delayed(single_bootstrap)(i) for i in range(n_iterations)
            )

            for i, (effect_size, ceiling) in enumerate(results):
                bootstrap_results['effect_sizes'][i] = effect_size
                bootstrap_results['ceiling_lines'][i] = ceiling

            bootstrap_results['statistics'] = {
                'effect_size_ci': np.percentile(bootstrap_results['effect_sizes'], [2.5, 97.5]).tolist(),
                'effect_size_mean': float(np.mean(bootstrap_results['effect_sizes'])),
                'effect_size_std': float(np.std(bootstrap_results['effect_sizes'])),
                'ceiling_ci_lower': np.percentile(bootstrap_results['ceiling_lines'], 2.5, axis=0),
                'ceiling_ci_upper': np.percentile(bootstrap_results['ceiling_lines'], 97.5, axis=0)
            }

            return bootstrap_results

        except Exception as e:
            raise ValueError(f"Error in bootstrap analysis: {str(e)}")

    def analyze(self, x: np.ndarray, y: np.ndarray, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Perform complete NCA analysis with comprehensive error handling."""
        try:
            # Input validation
            is_valid, errors = self.validate_inputs(x, y)
            if not is_valid:
                raise ValueError(f"Invalid input data: {', '.join(errors)}")

            # Calculate ceiling line based on method
            if self.method == 'CE-FDH':
                ceiling_y, method_stats = self.ce_fdh(x, y)
            elif self.method == 'CR-FDH':
                bandwidth = settings.get('bandwidth', 0.1)
                ceiling_y, method_stats = self.cr_fdh(x, y, bandwidth)
            else:
                quantile = settings.get('quantile', 0.95)
                ceiling_y, method_stats = self.quantile_regression(x, y, quantile)

            # Calculate effect size
            effect_size, effect_stats = self.calculate_effect_size(x, y, ceiling_y)

            # Perform bootstrap analysis
            bootstrap_results = self.bootstrap_analysis(x, y, settings.get('bootstrap_iterations', 1000))

            # Statistical tests
            test_results = self.perform_statistical_tests(x, y, ceiling_y)

            # Create bottleneck table
            bottleneck_table = self.compute_bottleneck_table(x, y, ceiling_y)

            # Compile results
            self.results = {
                'effect_size': effect_size,
                'ceiling_line': ceiling_y,
                'bottleneck_table': bottleneck_table,
                'bootstrap_results': bootstrap_results,
                'statistical_tests': test_results,
                'method_stats': method_stats,
                'effect_stats': effect_stats,
                'analysis_settings': settings,
                'data_summary': {
                    'n_observations': len(x),
                    'x_mean': float(np.mean(x)),
                    'x_std': float(np.std(x)),
                    'y_mean': float(np.mean(y)),
                    'y_std': float(np.std(y)),
                    'correlation': float(np.corrcoef(x, y)[0, 1])
                }
            }

            return self.results

        except Exception as e:
            raise ValueError(f"Error in NCA analysis: {str(e)}")

    def perform_statistical_tests(self, x: np.ndarray, y: np.ndarray, ceiling_y: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive statistical tests for NCA validity."""
        test_results = {}

        try:
            # Permutation test
            original_effect, _ = self.calculate_effect_size(x, y, ceiling_y)
            permuted_effects = np.zeros(1000)

            for i in range(1000):
                y_perm = np.random.permutation(y)
                if self.method == 'CE-FDH':
                    ceiling_perm, _ = self.ce_fdh(x, y_perm)
                elif self.method == 'CR-FDH':
                    ceiling_perm, _ = self.cr_fdh(x, y_perm)
                else:
                    ceiling_perm, _ = self.quantile_regression(x, y_perm)

                permuted_effects[i], _ = self.calculate_effect_size(x, y_perm, ceiling_perm)

            p_value = np.mean(permuted_effects >= original_effect)

            # Residuals analysis
            residuals = y - ceiling_y
            normality_stat, normality_p = stats.normaltest(residuals)

            test_results.update({
                'permutation_p_value': float(p_value),
                'normality_test_stat': float(normality_stat),
                'normality_test_p': float(normality_p),
                'residual_mean': float(np.mean(residuals)),
                'residual_std': float(np.std(residuals)),
                'residual_skewness': float(stats.skew(residuals)),
                'residual_kurtosis': float(stats.kurtosis(residuals))
            })

            return test_results

        except Exception as e:
            raise ValueError(f"Error in statistical testing: {str(e)}")

    def compute_bottleneck_table(self, x: np.ndarray, y: np.ndarray, ceiling_y: np.ndarray) -> pd.DataFrame:
        """Create detailed bottleneck analysis table."""
        try:
            percentiles = np.arange(0, 101, 5)
            x_levels = np.percentile(x, percentiles)
            y_levels = np.percentile(ceiling_y, percentiles)

            # Create base table
            bottleneck_df = pd.DataFrame({
                'Y Level (%)': percentiles,
                'Required X Level': x_levels,
                'Ceiling Y Level': y_levels,
                'X Level (Standardized)': (x_levels - np.mean(x)) / np.std(x),
                'Y Level (Standardized)': (y_levels - np.mean(y)) / np.std(y),
                'Sample Size': len(x)
            })

            # Calculate bottleneck identification
            bottleneck_df['Slope'] = bottleneck_df['Required X Level'].diff() / bottleneck_df['Y Level (%)'].diff()
            bottleneck_df['Is Bottleneck'] = False

            # First point is a bottleneck if it represents a significant constraint
            bottleneck_df.loc[0, 'Is Bottleneck'] = True

            # Identify bottlenecks based on slope changes and constraint level
            for i in range(1, len(bottleneck_df)):
                current_slope = bottleneck_df.loc[i, 'Slope']
                if pd.notnull(current_slope):
                    slope_change = abs(current_slope - bottleneck_df.loc[i - 1, 'Slope'])
                    y_level = bottleneck_df.loc[i, 'Y Level (%)']

                    bottleneck_df.loc[i, 'Is Bottleneck'] = (
                        slope_change > 0.1 and
                        y_level >= 20 and
                        y_level <= 80
                    )

            # Drop temporary slope column
            bottleneck_df.drop('Slope', axis=1, inplace=True)

            return bottleneck_df

        except Exception as e:
            raise ValueError(f"Error in bottleneck table computation: {str(e)}")
