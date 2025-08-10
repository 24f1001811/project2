import aiohttp
import asyncio
import pandas as pd
from bs4 import BeautifulSoup
import logging
from typing import Dict, List, Any
import re
from urllib.parse import urljoin, urlparse
import json

logger = logging.getLogger(__name__)

class WebScraper:
    """Handle web scraping and data extraction from URLs"""
    
    def __init__(self):
        self.session = None
        self.timeout = aiohttp.ClientTimeout(total=60)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    async def scrape_data(self, urls: List[str]) -> Dict[str, pd.DataFrame]:
        """Scrape data from list of URLs"""
        scraped_data = {}
        
        async with aiohttp.ClientSession(timeout=self.timeout, headers=self.headers) as session:
            self.session = session
            
            for i, url in enumerate(urls):
                try:
                    logger.info(f"Scraping URL: {url}")
                    df = await self._scrape_single_url(url)
                    scraped_data[f"scraped_{i+1}"] = df
                    logger.info(f"Successfully scraped {len(df)} rows from {url}")
                except Exception as e:
                    logger.error(f"Failed to scrape {url}: {str(e)}")
                    # Create empty DataFrame on failure
                    scraped_data[f"scraped_{i+1}"] = pd.DataFrame()
        
        return scraped_data
    
    async def _scrape_single_url(self, url: str) -> pd.DataFrame:
        """Scrape data from a single URL"""
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status} error")
                
                content = await response.text()
                content_type = response.headers.get('content-type', '').lower()
                
                # Handle different content types
                if 'application/json' in content_type:
                    return self._parse_json_response(content)
                elif 'text/csv' in content_type:
                    return self._parse_csv_response(content)
                else:
                    # Assume HTML and parse it
                    return self._parse_html_content(content, url)
                    
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            raise
    
    def _parse_json_response(self, content: str) -> pd.DataFrame:
        """Parse JSON response into DataFrame"""
        try:
            data = json.loads(content)
            
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                # Look for common data keys
                for key in ['data', 'results', 'records', 'items']:
                    if key in data and isinstance(data[key], list):
                        return pd.DataFrame(data[key])
                
                # If no list found, treat as single record
                return pd.DataFrame([data])
            else:
                return pd.DataFrame([{'value': data}])
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            return pd.DataFrame()
    
    def _parse_csv_response(self, content: str) -> pd.DataFrame:
        """Parse CSV response into DataFrame"""
        try:
            from io import StringIO
            return pd.read_csv(StringIO(content))
        except Exception as e:
            logger.error(f"CSV parsing error: {str(e)}")
            return pd.DataFrame()
    
    def _parse_html_content(self, content: str, url: str) -> pd.DataFrame:
        """Parse HTML content and extract structured data"""
        soup = BeautifulSoup(content, 'html.parser')
        
        # Try multiple extraction methods
        data_frames = []
        
        # Method 1: Look for tables
        tables_df = self._extract_tables(soup)
        if not tables_df.empty:
            data_frames.append(tables_df)
        
        # Method 2: Look for JSON-LD structured data
        json_ld_df = self._extract_json_ld(soup)
        if not json_ld_df.empty:
            data_frames.append(json_ld_df)
        
        # Method 3: Look for microdata
        microdata_df = self._extract_microdata(soup)
        if not microdata_df.empty:
            data_frames.append(microdata_df)
        
        # Method 4: Extract list items
        lists_df = self._extract_lists(soup)
        if not lists_df.empty:
            data_frames.append(lists_df)
        
        # Method 5: Extract text content with patterns
        text_df = self._extract_text_patterns(soup, url)
        if not text_df.empty:
            data_frames.append(text_df)
        
        # Return the largest DataFrame found
        if data_frames:
            largest_df = max(data_frames, key=len)
            logger.info(f"Extracted {len(largest_df)} rows from HTML")
            return largest_df
        
        # If no structured data found, return basic page info
        return pd.DataFrame([{
            'url': url,
            'title': soup.title.string if soup.title else 'No Title',
            'text_content': soup.get_text()[:1000]  # First 1000 chars
        }])
    
    def _extract_tables(self, soup: BeautifulSoup) -> pd.DataFrame:
        """Extract data from HTML tables"""
        tables = soup.find_all('table')
        
        if not tables:
            return pd.DataFrame()
        
        # Find the largest table
        largest_table = max(tables, key=lambda t: len(t.find_all('tr')))
        
        try:
            # Use pandas to read HTML tables
            import io
            table_html = str(largest_table)
            dfs = pd.read_html(io.StringIO(table_html))
            if dfs:
                return dfs[0]  # Return first table
        except:
            # Manual table parsing if pandas fails
            rows = largest_table.find_all('tr')
            if not rows:
                return pd.DataFrame()
            
            # Extract headers
            header_row = rows[0]
            headers = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
            
            # Extract data rows
            data_rows = []
            for row in rows[1:]:
                cells = [td.get_text().strip() for td in row.find_all(['td', 'th'])]
                if len(cells) == len(headers):
                    data_rows.append(cells)
            
            if data_rows:
                return pd.DataFrame(data_rows, columns=headers)
        
        return pd.DataFrame()
    
    def _extract_json_ld(self, soup: BeautifulSoup) -> pd.DataFrame:
        """Extract JSON-LD structured data"""
        json_scripts = soup.find_all('script', {'type': 'application/ld+json'})
        
        all_data = []
        for script in json_scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
            except:
                continue
        
        if all_data:
            return pd.DataFrame(all_data)
        
        return pd.DataFrame()
    
    def _extract_microdata(self, soup: BeautifulSoup) -> pd.DataFrame:
        """Extract microdata from HTML"""
        items = soup.find_all(attrs={'itemscope': True})
        
        data_rows = []
        for item in items:
            row_data = {}
            
            # Extract itemtype
            itemtype = item.get('itemtype')
            if itemtype:
                row_data['itemtype'] = itemtype
            
            # Extract properties
            props = item.find_all(attrs={'itemprop': True})
            for prop in props:
                prop_name = prop.get('itemprop')
                prop_value = prop.get('content') or prop.get_text().strip()
                row_data[prop_name] = prop_value
            
            if len(row_data) > 1:  # More than just itemtype
                data_rows.append(row_data)
        
        if data_rows:
            return pd.DataFrame(data_rows)
        
        return pd.DataFrame()
    
    def _extract_lists(self, soup: BeautifulSoup) -> pd.DataFrame:
        """Extract data from HTML lists"""
        lists = soup.find_all(['ul', 'ol'])
        
        all_items = []
        for lst in lists:
            items = lst.find_all('li')
            for item in items:
                text = item.get_text().strip()
                if text:
                    all_items.append({'item': text})
        
        if len(all_items) > 5:  # Only return if we have substantial data
            return pd.DataFrame(all_items)
        
        return pd.DataFrame()
    
    def _extract_text_patterns(self, soup: BeautifulSoup, url: str) -> pd.DataFrame:
        """Extract structured data from text patterns"""
        text = soup.get_text()
        
        # Look for common data patterns
        patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'price': r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
            'percentage': r'\d+(?:\.\d+)?%',
        }
        
        extracted_data = []
        for pattern_name, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                extracted_data.append({
                    'url': url,
                    'type': pattern_name,
                    'value': match
                })
        
        if extracted_data:
            return pd.DataFrame(extracted_data)
        
        return pd.DataFrame()