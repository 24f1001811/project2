import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import logging
from typing import Dict, Any, Union, List
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class StatisticalAnalyzer:
    """Perform various statistical analyses on datasets"""
    
    def __init__(self):
        self.analysis_methods = {
            'correlation': self._calculate_correlation,
            'regression': self._perform_regression,
            'mean': self._calculate_mean,
            'median': self._calculate_median,
            'std': self._calculate_std,
            'sum': self._calculate_sum,
            'count': self._calculate_count,
            'min': self._calculate_min,
            'max': self._calculate_max,
            'slope': self._calculate_slope,
            'r_squared': self._calculate_r_squared,
            'date_diff': self._calculate_date_diff,
            'percentage': self._calculate_percentage,
            'summary': self._calculate_summary,
            'basic_stats': self._calculate_summary,
        }
    
    def perform_analysis(self, datasets: Dict[str, pd.DataFrame], 
                        analysis_type: str, parameters: Dict[str, Any]) -> Any:
        """Perform statistical analysis on the datasets"""
        
        logger.info(f"Performing {analysis_type} analysis")
        
        if analysis_type not in self.analysis_methods:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
        
        # Combine all datasets if multiple exist
        combined_df = self._combine_datasets(datasets)
        
        if combined_df.empty:
            logger.warning("No data available for analysis")
            return None
        
        try:
            result = self.analysis_methods[analysis_type](combined_df, parameters)
            logger.info(f"{analysis_type} analysis completed")
            return result
        except Exception as e:
            logger.error(f"Error in {analysis_type} analysis: {str(e)}")
            raise
    
    def _combine_datasets(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple datasets into one"""
        if not datasets:
            return pd.DataFrame()
        
        if len(datasets) == 1:
            return list(datasets.values())[0]
        
        # Try to concatenate datasets with similar structures
        dfs = list(datasets.values())
        non_empty_dfs = [df for df in dfs if not df.empty]
        
        if not non_empty_dfs:
            return pd.DataFrame()
        
        try:
            # Try to concatenate by columns (if they have similar columns)
            combined = pd.concat(non_empty_dfs, ignore_index=True, sort=False)
            return combined
        except:
            # If concatenation fails, return the largest dataset
            return max(non_empty_dfs, key=len)
    
    def _find_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Find numeric columns in DataFrame"""
        return df.select_dtypes(include=[np.number]).columns.tolist()
    
    def _find_date_columns(self, df: pd.DataFrame) -> List[str]:
        """Find date columns in DataFrame"""
        date_cols = []
        for col in df.columns:
            if df[col].dtype in ['datetime64[ns]', 'datetime64']:
                date_cols.append(col)
            elif df[col].dtype == 'object':
                # Try to parse as datetime
                try:
                    pd.to_datetime(df[col].dropna().iloc[:5])  # Test first 5 non-null values
                    date_cols.append(col)
                except:
                    pass
        return date_cols
    
    def _calculate_correlation(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate correlation between columns"""
        numeric_cols = self._find_numeric_columns(df)
        
        if len(numeric_cols) < 2:
            return {"error": "Need at least 2 numeric columns for correlation"}
        
        # Use specified columns or all numeric columns
        if 'x_column' in params and 'y_column' in params:
            x_col = params['x_column']
            y_col = params['y_column']
            
            if x_col not in df.columns or y_col not in df.columns:
                return {"error": f"Specified columns not found"}
            
            correlation = df[x_col].corr(df[y_col])
            return {
                "correlation": float(correlation),
                "x_column": x_col,
                "y_column": y_col
            }
        else:
            # Calculate correlation matrix for all numeric columns
            corr_matrix = df[numeric_cols].corr()
            return {
                "correlation_matrix": corr_matrix.to_dict(),
                "columns": numeric_cols
            }
    
    def _perform_regression(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform linear regression"""
        numeric_cols = self._find_numeric_columns(df)
        
        if len(numeric_cols) < 2:
            return {"error": "Need at least 2 numeric columns for regression"}
        
        # Determine X and Y columns
        if 'x_column' in params and 'y_column' in params:
            x_col = params['x_column']
            y_col = params['y_column']
        else:
            x_col = numeric_cols[0]
            y_col = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
        
        if x_col not in df.columns or y_col not in df.columns:
            return {"error": "Regression columns not found"}
        
        # Clean data
        clean_df = df[[x_col, y_col]].dropna()
        
        if len(clean_df) < 2:
            return {"error": "Not enough valid data points for regression"}
        
        X = clean_df[x_col].values.reshape(-1, 1)
        y = clean_df[y_col].values
        
        # Perform regression
        model = LinearRegression()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        r_squared = r2_score(y, y_pred)
        
        return {
            "slope": float(model.coef_[0]),
            "intercept": float(model.intercept_),
            "r_squared": float(r_squared),
            "x_column": x_col,
            "y_column": y_col,
            "n_points": len(clean_df)
        }
    
    def _calculate_mean(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate mean of numeric columns"""
        numeric_cols = self._find_numeric_columns(df)
        
        if not numeric_cols:
            return {"error": "No numeric columns found"}
        
        means = {}
        for col in numeric_cols:
            means[col] = float(df[col].mean())
        
        return {"means": means}
    
    def _calculate_median(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate median of numeric columns"""
        numeric_cols = self._find_numeric_columns(df)
        
        if not numeric_cols:
            return {"error": "No numeric columns found"}
        
        medians = {}
        for col in numeric_cols:
            medians[col] = float(df[col].median())
        
        return {"medians": medians}
    
    def _calculate_std(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate standard deviation of numeric columns"""
        numeric_cols = self._find_numeric_columns(df)
        
        if not numeric_cols:
            return {"error": "No numeric columns found"}
        
        stds = {}
        for col in numeric_cols:
            stds[col] = float(df[col].std())
        
        return {"standard_deviations": stds}
    
    def _calculate_sum(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate sum of numeric columns"""
        numeric_cols = self._find_numeric_columns(df)
        
        if not numeric_cols:
            return {"error": "No numeric columns found"}
        
        sums = {}
        for col in numeric_cols:
            sums[col] = float(df[col].sum())
        
        return {"sums": sums}
    
    def _calculate_count(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate count of non-null values"""
        counts = {}
        for col in df.columns:
            counts[col] = int(df[col].count())
        
        return {"counts": counts, "total_rows": len(df)}
    
    def _calculate_min(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate minimum of numeric columns"""
        numeric_cols = self._find_numeric_columns(df)
        
        if not numeric_cols:
            return {"error": "No numeric columns found"}
        
        mins = {}
        for col in numeric_cols:
            mins[col] = float(df[col].min())
        
        return {"minimums": mins}
    
    def _calculate_max(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate maximum of numeric columns"""
        numeric_cols = self._find_numeric_columns(df)
        
        if not numeric_cols:
            return {"error": "No numeric columns found"}
        
        maxs = {}
        for col in numeric_cols:
            maxs[col] = float(df[col].max())
        
        return {"maximums": maxs}
    
    def _calculate_slope(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate regression slope"""
        regression_result = self._perform_regression(df, params)
        if "error" in regression_result:
            return regression_result
        
        return {"slope": regression_result["slope"]}
    
    def _calculate_r_squared(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate R-squared value"""
        regression_result = self._perform_regression(df, params)
        if "error" in regression_result:
            return regression_result
        
        return {"r_squared": regression_result["r_squared"]}
    
    def _calculate_date_diff(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate differences between dates"""
        date_cols = self._find_date_columns(df)
        
        if len(date_cols) < 1:
            return {"error": "No date columns found"}
        
        if len(date_cols) >= 2:
            # Calculate difference between first two date columns
            col1, col2 = date_cols[0], date_cols[1]
            
            # Convert to datetime
            df[col1] = pd.to_datetime(df[col1])
            df[col2] = pd.to_datetime(df[col2])
            
            diff_days = (df[col2] - df[col1]).dt.days
            
            return {
                "date_differences_days": diff_days.tolist(),
                "mean_difference_days": float(diff_days.mean()),
                "column1": col1,
                "column2": col2
            }
        else:
            # Calculate time spans within single date column
            date_col = date_cols[0]
            df[date_col] = pd.to_datetime(df[date_col])
            
            min_date = df[date_col].min()
            max_date = df[date_col].max()
            span_days = (max_date - min_date).days
            
            return {
                "date_span_days": span_days,
                "min_date": min_date.isoformat(),
                "max_date": max_date.isoformat(),
                "column": date_col
            }
    
    def _calculate_percentage(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate percentages and distributions"""
        results = {}
        
        # For categorical columns, calculate value distributions
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            percentages = (value_counts / len(df) * 100).round(2)
            results[col] = percentages.to_dict()
        
        # For numeric columns, calculate quartiles
        numeric_cols = self._find_numeric_columns(df)
        for col in numeric_cols:
            quartiles = df[col].quantile([0.25, 0.5, 0.75]).to_dict()
            results[f"{col}_quartiles"] = quartiles
        
        return {"percentages": results}
    
    def _calculate_summary(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics"""
        summary = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict()
        }
        
        # Numeric summary
        numeric_cols = self._find_numeric_columns(df)
        if numeric_cols:
            numeric_summary = df[numeric_cols].describe().to_dict()
            summary["numeric_summary"] = numeric_summary
        
        # Categorical summary
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            categorical_summary = {}
            for col in categorical_cols:
                categorical_summary[col] = {
                    "unique_count": df[col].nunique(),
                    "most_common": df[col].mode().iloc[0] if not df[col].mode().empty else None,
                    "null_count": df[col].isnull().sum()
                }
            summary["categorical_summary"] = categorical_summary
        
        return summary