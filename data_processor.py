import pandas as pd
import json
import io
import logging
from typing import Dict, Any, Union
import pyarrow.parquet as pq
import PyPDF2
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handle loading and processing of various file formats"""
    
    def __init__(self):
        self.supported_formats = {'.csv', '.json', '.parquet', '.pdf'}
    
    def load_data(self, filename: str, file_content: bytes) -> pd.DataFrame:
        """Load data from various file formats"""
        file_extension = self._get_file_extension(filename)
        
        logger.info(f"Loading {file_extension} file: {filename}")
        
        try:
            if file_extension == '.csv':
                return self._load_csv(file_content)
            elif file_extension == '.json':
                return self._load_json(file_content)
            elif file_extension == '.parquet':
                return self._load_parquet(file_content)
            elif file_extension == '.pdf':
                return self._load_pdf(file_content)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            raise
    
    def _get_file_extension(self, filename: str) -> str:
        """Get file extension in lowercase"""
        return '.' + filename.split('.')[-1].lower()
    
    def _load_csv(self, file_content: bytes) -> pd.DataFrame:
        """Load CSV file with robust parsing"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    content_str = file_content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not decode CSV file with any encoding")
            
            # Try different separators
            separators = [',', ';', '\t', '|']
            
            for sep in separators:
                try:
                    df = pd.read_csv(io.StringIO(content_str), sep=sep)
                    if len(df.columns) > 1:  # Valid if more than one column
                        # Clean column names
                        df.columns = df.columns.str.strip()
                        return self._clean_dataframe(df)
                except:
                    continue
            
            # If all else fails, try with default settings
            df = pd.read_csv(io.StringIO(content_str))
            return self._clean_dataframe(df)
            
        except Exception as e:
            logger.error(f"CSV loading error: {str(e)}")
            raise
    
    def _load_json(self, file_content: bytes) -> pd.DataFrame:
        """Load JSON file"""
        try:
            content_str = file_content.decode('utf-8')
            data = json.loads(content_str)
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                elif 'records' in data:
                    df = pd.DataFrame(data['records'])
                else:
                    # Try to create DataFrame from dict values
                    df = pd.DataFrame([data])
            else:
                raise ValueError("JSON format not supported")
            
            return self._clean_dataframe(df)
            
        except Exception as e:
            logger.error(f"JSON loading error: {str(e)}")
            raise
    
    def _load_parquet(self, file_content: bytes) -> pd.DataFrame:
        """Load Parquet file"""
        try:
            df = pd.read_parquet(io.BytesIO(file_content))
            return self._clean_dataframe(df)
        except Exception as e:
            logger.error(f"Parquet loading error: {str(e)}")
            raise
    
    def _load_pdf(self, file_content: bytes) -> pd.DataFrame:
        """Extract text from PDF and try to create structured data"""
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""
            
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            # Try to extract tabular data from text
            lines = text.strip().split('\n')
            
            # Look for patterns that might be tabular
            potential_data = []
            headers = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Try to split by common delimiters
                for delimiter in ['\t', '  ', '|', ',']:
                    parts = [part.strip() for part in line.split(delimiter) if part.strip()]
                    if len(parts) > 1:
                        if headers is None and not any(char.isdigit() for char in line):
                            headers = parts
                        else:
                            potential_data.append(parts)
                        break
            
            if headers and potential_data:
                # Ensure all rows have the same length as headers
                cleaned_data = []
                for row in potential_data:
                    if len(row) == len(headers):
                        cleaned_data.append(row)
                
                if cleaned_data:
                    df = pd.DataFrame(cleaned_data, columns=headers)
                    return self._clean_dataframe(df)
            
            # If no tabular data found, create a single-column DataFrame with text
            df = pd.DataFrame({'text': [text]})
            return df
            
        except Exception as e:
            logger.error(f"PDF loading error: {str(e)}")
            raise
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize DataFrame"""
        try:
            # Remove completely empty rows
            df = df.dropna(how='all')
            
            # Clean column names
            df.columns = df.columns.astype(str).str.strip()
            
            # Try to infer better data types
            for col in df.columns:
                # Try to convert to numeric
                if df[col].dtype == 'object':
                    # Remove common non-numeric characters
                    cleaned_series = df[col].astype(str).str.replace(r'[$,\s]', '', regex=True)
                    
                    # Try to convert to numeric
                    numeric_series = pd.to_numeric(cleaned_series, errors='ignore')
                    if not numeric_series.equals(cleaned_series):
                        df[col] = numeric_series
                    
                    # Try to convert to datetime
                    if df[col].dtype == 'object':
                        try:
                            datetime_series = pd.to_datetime(df[col], errors='ignore', infer_datetime_format=True)
                            if not datetime_series.equals(df[col]):
                                df[col] = datetime_series
                        except:
                            pass
            
            logger.info(f"DataFrame loaded with shape {df.shape} and columns: {list(df.columns)}")
            return df
            
        except Exception as e:
            logger.error(f"DataFrame cleaning error: {str(e)}")
            return df  # Return uncleaned if cleaning fails