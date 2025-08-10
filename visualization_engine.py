import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import base64
import io
from typing import Dict, Any, List
import logging
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')
plt.switch_backend('Agg')  # Use non-interactive backend
logger = logging.getLogger(__name__)

class VisualizationEngine:
    """Create data visualizations and encode them as base64 images"""
    
    def __init__(self):
        self.plot_methods = {
            'scatter': self._create_scatter_plot,
            'line': self._create_line_plot,
            'bar': self._create_bar_plot,
            'histogram': self._create_histogram,
            'plot': self._create_auto_plot,
            'chart': self._create_auto_plot,
            'graph': self._create_auto_plot,
            'visualiz': self._create_auto_plot,
            'regression_line': self._create_regression_plot,
        }
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 100
    
    def create_plot(self, datasets: Dict[str, pd.DataFrame], 
                   plot_type: str, parameters: Dict[str, Any]) -> str:
        """Create a plot and return as base64 encoded string"""
        
        logger.info(f"Creating {plot_type} plot")
        
        # Combine datasets
        combined_df = self._combine_datasets(datasets)
        
        if combined_df.empty:
            logger.warning("No data available for plotting")
            return self._create_error_plot("No data available")
        
        if plot_type not in self.plot_methods:
            plot_type = 'plot'  # Default to auto plot
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            self.plot_methods[plot_type](combined_df, ax, parameters)
            
            # Apply common formatting
            self._apply_formatting(ax, parameters)
            
            # Convert to base64
            base64_image = self._fig_to_base64(fig)
            plt.close(fig)
            
            # Check size constraint (100KB)
            if len(base64_image) > 100000:
                logger.warning("Plot size exceeds 100KB, creating smaller version")
                # Create smaller version
                fig, ax = plt.subplots(figsize=(8, 5))
                self.plot_methods[plot_type](combined_df, ax, parameters)
                self._apply_formatting(ax, parameters)
                plt.tight_layout()
                base64_image = self._fig_to_base64(fig, quality=70)
                plt.close(fig)
            
            logger.info(f"Plot created successfully, size: {len(base64_image)} bytes")
            return f"data:image/png;base64,{base64_image}"
            
        except Exception as e:
            logger.error(f"Error creating plot: {str(e)}")
            plt.close('all')  # Clean up any open figures
            return self._create_error_plot(f"Plot creation failed: {str(e)}")
    
    def _combine_datasets(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple datasets into one"""
        if not datasets:
            return pd.DataFrame()
        
        if len(datasets) == 1:
            return list(datasets.values())[0]
        
        # Try to concatenate datasets
        dfs = [df for df in datasets.values() if not df.empty]
        if not dfs:
            return pd.DataFrame()
        
        try:
            combined = pd.concat(dfs, ignore_index=True, sort=False)
            return combined
        except:
            return max(dfs, key=len)  # Return largest dataset
    
    def _find_plot_columns(self, df: pd.DataFrame, params: Dict[str, Any]) -> tuple:
        """Find appropriate columns for plotting"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Use specified columns if available
        x_col = params.get('x_column')
        y_col = params.get('y_column')
        
        if x_col and x_col in df.columns:
            x_column = x_col
        else:
            # Auto-select X column (prefer first numeric, then any column)
            x_column = numeric_cols[0] if numeric_cols else df.columns[0]
        
        if y_col and y_col in df.columns:
            y_column = y_col
        else:
            # Auto-select Y column (prefer second numeric, then first numeric)
            if len(numeric_cols) > 1:
                y_column = numeric_cols[1]
            elif len(numeric_cols) > 0:
                y_column = numeric_cols[0]
            else:
                y_column = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        return x_column, y_column
    
    def _create_scatter_plot(self, df: pd.DataFrame, ax, params: Dict[str, Any]):
        """Create scatter plot"""
        x_col, y_col = self._find_plot_columns(df, params)
        
        # Clean data
        clean_df = df[[x_col, y_col]].dropna()
        
        if clean_df.empty:
            ax.text(0.5, 0.5, 'No valid data to plot', 
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        # Create scatter plot
        ax.scatter(clean_df[x_col], clean_df[y_col], alpha=0.6, s=50)
        
        # Add regression line if requested
        if params.get('show_regression', False):
            self._add_regression_line(clean_df, x_col, y_col, ax)
        
        ax.set_xlabel(params.get('xlabel', x_col))
        ax.set_ylabel(params.get('ylabel', y_col))
    
    def _create_line_plot(self, df: pd.DataFrame, ax, params: Dict[str, Any]):
        """Create line plot"""
        x_col, y_col = self._find_plot_columns(df, params)
        
        # Clean data and sort by x
        clean_df = df[[x_col, y_col]].dropna().sort_values(x_col)
        
        if clean_df.empty:
            ax.text(0.5, 0.5, 'No valid data to plot', 
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        ax.plot(clean_df[x_col], clean_df[y_col], marker='o', linewidth=2, markersize=4)
        ax.set_xlabel(params.get('xlabel', x_col))
        ax.set_ylabel(params.get('ylabel', y_col))
    
    def _create_bar_plot(self, df: pd.DataFrame, ax, params: Dict[str, Any]):
        """Create bar plot"""
        x_col, y_col = self._find_plot_columns(df, params)
        
        # If x is categorical, use value counts
        if df[x_col].dtype == 'object':
            if df[y_col].dtype in ['object']:
                # Both categorical - use value counts of x
                counts = df[x_col].value_counts().head(10)  # Limit to top 10
                ax.bar(range(len(counts)), counts.values)
                ax.set_xticks(range(len(counts)))
                ax.set_xticklabels(counts.index, rotation=45, ha='right')
                ax.set_ylabel('Count')
            else:
                # X categorical, Y numeric - group by X and average Y
                grouped = df.groupby(x_col)[y_col].mean().head(10)
                ax.bar(range(len(grouped)), grouped.values)
                ax.set_xticks(range(len(grouped)))
                ax.set_xticklabels(grouped.index, rotation=45, ha='right')
                ax.set_ylabel(params.get('ylabel', f'Mean {y_col}'))
        else:
            # Both numeric - create binned bar chart
            bins = min(20, len(df[x_col].unique()))
            ax.hist(df[x_col].dropna(), bins=bins, edgecolor='black', alpha=0.7)
            ax.set_ylabel('Frequency')
        
        ax.set_xlabel(params.get('xlabel', x_col))
    
    def _create_histogram(self, df: pd.DataFrame, ax, params: Dict[str, Any]):
        """Create histogram"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if not numeric_cols.empty:
            col = params.get('column', numeric_cols[0])
            if col not in df.columns:
                col = numeric_cols[0]
            
            data = df[col].dropna()
            if not data.empty:
                bins = min(30, len(data.unique()))
                ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)
                ax.set_xlabel(params.get('xlabel', col))
                ax.set_ylabel('Frequency')
            else:
                ax.text(0.5, 0.5, 'No valid data to plot', 
                       transform=ax.transAxes, ha='center', va='center')
        else:
            ax.text(0.5, 0.5, 'No numeric data available', 
                   transform=ax.transAxes, ha='center', va='center')
    
    def _create_auto_plot(self, df: pd.DataFrame, ax, params: Dict[str, Any]):
        """Automatically choose appropriate plot type"""
        x_col, y_col = self._find_plot_columns(df, params)
        
        # Decision logic for plot type
        if df[x_col].dtype == 'object' and df[y_col].dtype == 'object':
            # Both categorical - bar plot of counts
            self._create_bar_plot(df, ax, params)
        elif df[x_col].dtype == 'object':
            # X categorical, Y numeric - bar plot
            self._create_bar_plot(df, ax, params)
        elif len(df) > 100:
            # Large dataset - scatter plot
            self._create_scatter_plot(df, ax, params)
        else:
            # Small dataset - line plot
            self._create_line_plot(df, ax, params)
    
    def _create_regression_plot(self, df: pd.DataFrame, ax, params: Dict[str, Any]):
        """Create scatter plot with regression line"""
        params['show_regression'] = True
        self._create_scatter_plot(df, ax, params)
    
    def _add_regression_line(self, df: pd.DataFrame, x_col: str, y_col: str, ax):
        """Add regression line to existing plot"""
        try:
            X = df[x_col].values.reshape(-1, 1)
            y = df[y_col].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Create line
            x_line = np.linspace(df[x_col].min(), df[x_col].max(), 100)
            y_line = model.predict(x_line.reshape(-1, 1))
            
            ax.plot(x_line, y_line, color='red', linewidth=2, label='Regression Line')
            
            # Add R² to legend
            from sklearn.metrics import r2_score
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
        except Exception as e:
            logger.warning(f"Could not add regression line: {str(e)}")
    
    def _apply_formatting(self, ax, params: Dict[str, Any]):
        """Apply common formatting to the plot"""
        # Set title
        title = params.get('title', 'Data Visualization')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Apply tight layout
        plt.tight_layout()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels if they're long
        labels = ax.get_xticklabels()
        if labels and any(len(label.get_text()) > 10 for label in labels):
            ax.tick_params(axis='x', rotation=45)
    
    def _fig_to_base64(self, fig, quality=95) -> str:
        """Convert matplotlib figure to base64 string"""
        buffer = io.BytesIO()
        
        # Save with different qualities to control size
        if quality < 95:
            fig.savefig(buffer, format='png', dpi=80, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        else:
            fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        
        return image_base64
    
    def _create_error_plot(self, error_message: str) -> str:
        """Create an error plot when visualization fails"""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f'Plot Error:\n{error_message}', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title('Visualization Error', fontsize=14, fontweight='bold')
            
            base64_image = self._fig_to_base64(fig)
            plt.close(fig)
            
            return f"data:image/png;base64,{base64_image}"
            
        except Exception as e:
            logger.error(f"Could not create error plot: {str(e)}")
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="