# Data Analyst Agent API

A production-ready FastAPI service that automatically performs data analysis based on natural language questions and uploaded files. The API can handle web scraping, statistical analysis, and data visualization with automatic output formatting.

## 🚀 Quick Start

### Local Deployment

1. **Clone and setup:**
```bash
git clone <repository>
cd data-analyst-api
chmod +x deploy.sh
./deploy.sh
```

2. **Manual setup (alternative):**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

### Docker Deployment

```bash
docker build -t data-analyst-api .
docker run -p 8000:8000 data-analyst-api
```

### Public Access with ngrok

```bash
# Install ngrok first: https://ngrok.com/download
ngrok http 8000
# Use the provided public URL
```

## 📡 API Endpoints

### Main Analysis Endpoint
```
POST /api/analyze
```

**Required Files:**
- `questions.txt` (required) - Natural language description of the analysis task
- Additional data files (optional) - `.csv`, `.json`, `.parquet`, `.pdf`

**Response:** JSON object or array based on request requirements

### Health Check
```
GET /health
```

### API Information
```
GET /
```

## 🔥 Capabilities

### Data Sources
- **File Upload**: CSV, JSON, Parquet, PDF
- **Web Scraping**: Extract data from URLs (HTML tables, JSON APIs, etc.)
- **Multiple Formats**: Automatic format detection and parsing

### Statistical Analysis
- **Correlations**: Pearson correlation coefficients
- **Regression**: Linear regression with R² and slope
- **Descriptive Stats**: Mean, median, std, min, max, count
- **Date Operations**: Date differences and time spans
- **Percentages**: Distribution analysis and quartiles

### Data Visualization
- **Scatter Plots**: With optional regression lines
- **Line Charts**: Time series and trend analysis  
- **Bar Charts**: Categorical data visualization
- **Histograms**: Distribution visualization
- **Auto-Selection**: Intelligent plot type selection
- **Base64 Encoding**: Images under 100KB limit

### Output Formatting
- **JSON Objects**: Structured key-value results
- **JSON Arrays**: Ordered list results
- **Exact Formatting**: Matches requested output structure
- **Image Integration**: Base64 encoded plots in JSON

## 📝 Example Usage

### Example 1: Simple Analysis
**questions.txt:**
```
Calculate the correlation between sales and marketing_spend columns.
Return as JSON object with correlation value.
```

**Response:**
```json
{
  "correlation": {
    "correlation": 0.985,
    "x_column": "marketing_spend", 
    "y_column": "sales"
  }
}
```

### Example 2: Web Scraping + Visualization
**questions.txt:**
```
Scrape data from https://example.com/data-table
Create a scatter plot with regression line
Return plot as base64 image
```

**Response:**
```json
{
  "scatter_plot": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```

### Example 3: Array Output
**questions.txt:**
```
Calculate mean, median, and std of price column.
Return as JSON array: [mean, median, std]
```

**Response:**
```json
[45.2, 42.0, 12.8]
```

## 🧪 Testing

Run the test suite:
```bash
python test_client.py [API_URL]
```

Example test with custom URL:
```bash
python test_client.py https://your-api.ngrok.io
```

## 🏗️ Architecture

```
main.py                 # FastAPI application and routing
├── question_parser.py  # Natural language parsing
├── data_processor.py   # File loading and cleaning  
├── web_scraper.py      # URL scraping and extraction
├── statistical_analyzer.py # Statistical computations
└── visualization_engine.py # Plot generation
```

## 🔒 Production Features

- **3-minute timeout** for all analysis operations
- **Robust error handling** with detailed logging
- **Memory-efficient** processing for large datasets
- **Format validation** for uploads and outputs
- **Size constraints** for generated images
- **CORS enabled** for web applications
- **Health monitoring** endpoint
- **Docker support** for containerized deployment

## 🛠️ Configuration

### Environment Variables
```bash
HOST=0.0.0.0          # Server host (default: 0.0.0.0)
PORT=8000             # Server port (default: 8000)
```

### File Size Limits
- **Image outputs**: 100KB max (automatically compressed)
- **Upload files**: No hard limit (memory constrained)
- **Processing time**: 180 seconds max

## 🔍 Supported Question Patterns

The API recognizes these patterns in questions.txt:

- **Web scraping**: "scrape data from URL", "extract from https://..."
- **Statistics**: "correlation", "regression", "mean", "calculate slope"
- **Visualization**: "scatter plot", "create chart", "plot with regression line"
- **Output format**: "return as JSON array", "return as object"
- **Column specification**: "between column1 and column2", "x-axis: time"

## 🚨 Error Handling

The API provides structured error responses:
```json
{
  "detail": "Analysis timeout exceeded 3 minutes"
}
```

Common error scenarios:
- Missing questions.txt file
- Analysis timeout (>3 minutes)
- Invalid data formats
- Network errors during scraping
- Insufficient data for analysis

## 📊 Performance

- **Cold start**: ~2-3 seconds
- **Typical analysis**: 10-30 seconds  
- **Large datasets**: Up to 3 minutes (with timeout)
- **Concurrent requests**: Supported (async processing)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.