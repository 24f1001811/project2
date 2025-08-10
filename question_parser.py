import re
import json
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class QuestionParser:
    """Parse questions.txt to understand analysis requirements"""
    
    def __init__(self):
        self.scraping_patterns = [
            r'scrape?\s+(?:data\s+)?from\s+(https?://[^\s]+)',
            r'extract\s+(?:data\s+)?from\s+(https?://[^\s]+)',
            r'get\s+(?:data\s+)?from\s+(https?://[^\s]+)',
            r'fetch\s+(?:data\s+)?from\s+(https?://[^\s]+)',
        ]
        
        self.statistical_patterns = {
            'correlation': r'correlat(?:ion|e)',
            'regression': r'regress(?:ion|ive)',
            'mean': r'(?:average|mean)',
            'median': r'median',
            'std': r'standard\s+deviation|std',
            'sum': r'sum(?:mary)?',
            'count': r'count',
            'min': r'minim(?:um|al)',
            'max': r'maxim(?:um|al)',
            'slope': r'slope',
            'r_squared': r'r\^?2|r.squared',
            'date_diff': r'date\s+(?:difference|diff)',
            'percentage': r'percent(?:age)?',
        }
        
        self.visualization_patterns = {
            'scatter': r'scatter\s*plot',
            'line': r'line\s+(?:chart|plot)',
            'bar': r'bar\s+(?:chart|plot)',
            'histogram': r'histogram',
            'regression_line': r'regression\s+line',
            'plot': r'plot',
            'chart': r'chart',
            'graph': r'graph',
            'visualiz': r'visualiz',
        }
        
        self.output_format_patterns = {
            'array': r'(?:return\s+)?(?:as\s+)?(?:a\s+)?(?:json\s+)?array',
            'object': r'(?:return\s+)?(?:as\s+)?(?:a\s+)?(?:json\s+)?object',
            'list': r'(?:return\s+)?(?:as\s+)?(?:a\s+)?list',
        }
    
    def parse(self, questions_content: str) -> Dict[str, Any]:
        """Parse the questions content and return analysis plan"""
        content_lower = questions_content.lower()
        
        plan = {
            'requires_scraping': False,
            'scrape_urls': [],
            'statistical_tasks': [],
            'visualization_tasks': [],
            'output_format': 'object',
            'raw_content': questions_content
        }
        
        # Check for web scraping requirements
        urls = self._extract_urls(questions_content)
        if urls:
            plan['requires_scraping'] = True
            plan['scrape_urls'] = urls
        
        # Identify statistical analysis tasks
        plan['statistical_tasks'] = self._identify_statistical_tasks(content_lower)
        
        # Identify visualization tasks
        plan['visualization_tasks'] = self._identify_visualization_tasks(content_lower)
        
        # Determine output format
        plan['output_format'] = self._determine_output_format(content_lower)
        
        # Parse specific field requirements for arrays
        if plan['output_format'] == 'array':
            plan['array_fields'] = self._extract_array_fields(questions_content)
        
        return plan
    
    def _extract_urls(self, content: str) -> List[str]:
        """Extract URLs from the content"""
        urls = []
        for pattern in self.scraping_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            urls.extend(matches)
        
        # Also find standalone URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        standalone_urls = re.findall(url_pattern, content)
        urls.extend(standalone_urls)
        
        return list(set(urls))  # Remove duplicates
    
    def _identify_statistical_tasks(self, content_lower: str) -> List[Dict[str, Any]]:
        """Identify what statistical analysis is needed"""
        tasks = []
        
        for task_type, pattern in self.statistical_patterns.items():
            if re.search(pattern, content_lower):
                task_config = {
                    'name': task_type,
                    'type': task_type,
                    'parameters': self._extract_task_parameters(content_lower, task_type)
                }
                tasks.append(task_config)
        
        # If no specific stats mentioned but data processing implied, add basic stats
        if not tasks and any(word in content_lower for word in ['analyze', 'calculate', 'compute', 'find']):
            tasks.append({
                'name': 'basic_stats',
                'type': 'summary',
                'parameters': {}
            })
        
        return tasks
    
    def _identify_visualization_tasks(self, content_lower: str) -> List[Dict[str, Any]]:
        """Identify what visualizations are needed"""
        tasks = []
        
        for viz_type, pattern in self.visualization_patterns.items():
            if re.search(pattern, content_lower):
                task_config = {
                    'name': f'{viz_type}_plot',
                    'type': viz_type,
                    'parameters': self._extract_viz_parameters(content_lower, viz_type)
                }
                tasks.append(task_config)
                break  # Usually want one main visualization
        
        return tasks
    
    def _determine_output_format(self, content_lower: str) -> str:
        """Determine the required output format"""
        for format_type, pattern in self.output_format_patterns.items():
            if re.search(pattern, content_lower):
                return 'array' if format_type in ['array', 'list'] else 'object'
        
        return 'object'  # Default
    
    def _extract_array_fields(self, content: str) -> List[str]:
        """Extract field names for array output"""
        # Look for patterns like "return [field1, field2, field3]"
        array_match = re.search(r'\[([^\]]+)\]', content)
        if array_match:
            fields_str = array_match.group(1)
            fields = [field.strip().strip('"\'') for field in fields_str.split(',')]
            return fields
        
        return []
    
    def _extract_task_parameters(self, content_lower: str, task_type: str) -> Dict[str, Any]:
        """Extract parameters for statistical tasks"""
        params = {}
        
        # Extract column names
        column_patterns = [
            r'(?:column|field|variable)s?\s+["\']([^"\']+)["\']',
            r'between\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+and\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        ]
        
        for pattern in column_patterns:
            matches = re.findall(pattern, content_lower)
            if matches:
                if task_type in ['correlation', 'regression']:
                    if len(matches[0]) == 2:  # tuple from "between X and Y"
                        params['x_column'] = matches[0][0]
                        params['y_column'] = matches[0][1]
                break
        
        return params
    
    def _extract_viz_parameters(self, content_lower: str, viz_type: str) -> Dict[str, Any]:
        """Extract parameters for visualization tasks"""
        params = {}
        
        # Extract axis labels
        if 'x-axis' in content_lower or 'x axis' in content_lower:
            x_match = re.search(r'x[- ]axis[:\s]+["\']?([^"\'\\n,]+)["\']?', content_lower)
            if x_match:
                params['xlabel'] = x_match.group(1).strip()
        
        if 'y-axis' in content_lower or 'y axis' in content_lower:
            y_match = re.search(r'y[- ]axis[:\s]+["\']?([^"\'\\n,]+)["\']?', content_lower)
            if y_match:
                params['ylabel'] = y_match.group(1).strip()
        
        # Extract title
        title_match = re.search(r'title[:\s]+["\']?([^"\'\\n,]+)["\']?', content_lower)
        if title_match:
            params['title'] = title_match.group(1).strip()
        
        # Check if regression line needed
        if 'regression' in content_lower or 'trend' in content_lower:
            params['show_regression'] = True
        
        return params