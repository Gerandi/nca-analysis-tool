import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import IsolationForest

class DataProcessor:
    def __init__(self):
        self.data = None
        self.original_data = None
        self.processed_data = None
        self.preprocessing_stats = {}
        self.column_types = {}

    def prepare_data_for_analysis(self, x_var: str, y_var: str, settings: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Prepare data for NCA analysis with comprehensive preprocessing."""
        if self.data is None:
            raise ValueError("No data loaded")
            
        preprocessing_info = {
            'x_var': x_var,
            'y_var': y_var,
            'settings': settings,
            'steps': []
        }
        
        try:
            # Create working copy from original data
            self.processed_data = self.data[[x_var, y_var]].copy()
            
            # Basic data cleaning (always performed)
            # Replace inf/-inf with NaN
            self.processed_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            # Drop any NaN values
            initial_size = len(self.processed_data)
            self.processed_data.dropna(inplace=True)
            rows_removed = initial_size - len(self.processed_data)
            
            if rows_removed > 0:
                preprocessing_info['steps'].append({
                    'step': 'basic_cleaning',
                    'action': 'removed_invalid',
                    'rows_removed': rows_removed
                })
            
            # Basic data validation
            if len(self.processed_data) == 0:
                raise ValueError("All data was invalid after basic cleaning")
            
            # Apply optional preprocessing steps
            if settings:
                # Outlier detection
                if settings.get('outlier_method') not in [None, 'none']:
                    for var in [x_var, y_var]:
                        outliers, stats = self.detect_outliers(
                            self.processed_data[var],
                            method=settings['outlier_method'],
                            threshold=settings.get('outlier_threshold', 3.0)
                        )
                        self.processed_data = self.processed_data[~outliers]
                        preprocessing_info['steps'].append({
                            'step': 'outliers',
                            'variable': var,
                            'stats': stats
                        })

                # Scaling
                if settings.get('scaling_method') not in [None, 'none']:
                    for var in [x_var, y_var]:
                        scaled_data, scale_params = self.scale_data(
                            self.processed_data[var],
                            method=settings['scaling_method']
                        )
                        self.processed_data[var] = scaled_data
                        preprocessing_info['steps'].append({
                            'step': 'scaling',
                            'variable': var,
                            'params': scale_params
                        })

                # Transformation
                if settings.get('transform_method') not in [None, 'none']:
                    for var in [x_var, y_var]:
                        transformed_data, transform_params = self.transform_data(
                            self.processed_data[var],
                            method=settings['transform_method']
                        )
                        self.processed_data[var] = transformed_data
                        preprocessing_info['steps'].append({
                            'step': 'transformation',
                            'variable': var,
                            'params': transform_params
                        })

            # Final validation
            final_validation = self.processed_data[[x_var, y_var]].copy()
            if not np.all(np.isfinite(final_validation)):
                raise ValueError("Data contains non-finite values after preprocessing")
            
            preprocessing_info['final_sample_size'] = len(self.processed_data)
            return self.processed_data, preprocessing_info
            
        except Exception as e:
            raise ValueError(f"Error in data preparation: {str(e)}")

    def detect_outliers(self, data: pd.Series, method: str = 'zscore', threshold: float = 3.0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Detect outliers using various methods."""
        outlier_stats = {'method': method, 'threshold': threshold}
        data = pd.to_numeric(data, errors='coerce')
        data = data[np.isfinite(data)]  # Remove any inf values
        
        try:
            if method == 'zscore':
                z_scores = np.abs(stats.zscore(data))
                outliers = z_scores > threshold
            elif method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))
            elif method == 'isolation_forest':
                iso = IsolationForest(contamination=0.1, random_state=42)
                outliers = iso.fit_predict(data.values.reshape(-1, 1)) == -1
            else:
                outliers = np.zeros(len(data), dtype=bool)
            
            outlier_stats.update({
                'outliers_found': int(np.sum(outliers)),
                'percentage': float(np.sum(outliers) / len(data) * 100)
            })
            return outliers, outlier_stats
            
        except Exception as e:
            raise ValueError(f"Error in outlier detection: {str(e)}")

    def scale_data(self, data: pd.Series, method: str = 'standard') -> Tuple[pd.Series, Dict[str, Any]]:
        """Scale data using various methods."""
        if method in ['none', None]:
            return data, {'method': 'none'}
            
        data = pd.to_numeric(data, errors='coerce')
        data = data[np.isfinite(data)]
        
        try:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                return data, {'method': 'none'}
                
            scaled_data = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
            return pd.Series(scaled_data, index=data.index), {'method': method}
            
        except Exception as e:
            raise ValueError(f"Error in scaling: {str(e)}")

    def transform_data(self, data: pd.Series, method: str = None) -> Tuple[pd.Series, Dict[str, Any]]:
        """Transform data using various methods."""
        if method in ['none', None]:
            return data, {'method': 'none'}
            
        data = pd.to_numeric(data, errors='coerce')
        data = data[np.isfinite(data)]
        
        try:
            if method == 'log':
                # Handle negative values
                min_val = data.min()
                if min_val <= 0:
                    offset = abs(min_val) + 1
                    data = data + offset
                transformed = np.log1p(data)
            elif method == 'sqrt':
                # Handle negative values
                min_val = data.min()
                if min_val < 0:
                    offset = abs(min_val)
                    data = data + offset
                transformed = np.sqrt(data)
            elif method == 'box-cox':
                # Box-Cox requires positive values
                min_val = data.min()
                if min_val <= 0:
                    offset = abs(min_val) + 1
                    data = data + offset
                transformed, _ = stats.boxcox(data)
            else:
                return data, {'method': 'none'}
                
            return pd.Series(transformed, index=data.index), {'method': method}
            
        except Exception as e:
            raise ValueError(f"Error in transformation: {str(e)}")

    def load_data(self, file) -> Tuple[Optional[pd.DataFrame], str]:
        """Load data from file."""
        try:
            df = pd.read_csv(file)
            
            # Basic validation
            if df.empty:
                return None, "The uploaded file is empty"
                
            # Store the data
            self.data = df
            self.original_data = df.copy()
            
            return df, "Data loaded successfully"
            
        except Exception as e:
            return None, f"Error loading data: {str(e)}"