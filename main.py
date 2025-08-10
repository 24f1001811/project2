from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import logging
import os
import tempfile
import traceback
from typing import List, Dict, Any, Optional, Union
import uvicorn
from datetime import datetime
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Data Analyst Agent API",
    description="AI-powered data analysis API that processes questions and files",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import analysis modules
from data_processor import DataProcessor
from web_scraper import WebScraper
from statistical_analyzer import StatisticalAnalyzer
from visualization_engine import VisualizationEngine
from question_parser import QuestionParser

class AnalysisTimeout(Exception):
    """Custom exception for analysis timeout"""
    pass

class DataAnalystAgent:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.web_scraper = WebScraper()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.visualization_engine = VisualizationEngine()
        self.question_parser = QuestionParser()
        
    async def process_request(self, questions_content: str, files: Dict[str, bytes]) -> Dict[str, Any]:
        """Process the analysis request with timeout"""
        try:
            # Set up timeout handler
            def timeout_handler(signum, frame):
                raise AnalysisTimeout("Analysis exceeded 3-minute timeout")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(180)  # 3 minutes timeout
            
            try:
                result = await self._perform_analysis(questions_content, files)
                signal.alarm(0)  # Cancel timeout
                return result
            except AnalysisTimeout:
                signal.alarm(0)
                raise HTTPException(status_code=408, detail="Analysis timeout exceeded 3 minutes")
                
        except Exception as e:
            signal.alarm(0)  # Ensure timeout is cancelled
            logger.error(f"Analysis error: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    async def _perform_analysis(self, questions_content: str, files: Dict[str, bytes]) -> Dict[str, Any]:
        """Perform the actual analysis"""
        logger.info("Starting analysis process")
        
        # Parse the question to understand what's needed
        analysis_plan = self.question_parser.parse(questions_content)
        logger.info(f"Analysis plan: {analysis_plan}")
        
        # Initialize data storage
        datasets = {}
        
        # Process uploaded files
        for filename, file_content in files.items():
            if filename.lower().endswith(('.csv', '.json', '.parquet')):
                datasets[filename] = self.data_processor.load_data(filename, file_content)
                logger.info(f"Loaded dataset: {filename}")
        
        # Handle web scraping if needed
        if analysis_plan.get('requires_scraping'):
            scraped_data = await self.web_scraper.scrape_data(analysis_plan.get('scrape_urls', []))
            datasets.update(scraped_data)
        
        # Perform statistical analysis
        results = {}
        if analysis_plan.get('statistical_tasks'):
            for task in analysis_plan['statistical_tasks']:
                task_result = self.statistical_analyzer.perform_analysis(
                    datasets, task['type'], task.get('parameters', {})
                )
                results[task['name']] = task_result
        
        # Generate visualizations if needed
        if analysis_plan.get('visualization_tasks'):
            for viz_task in analysis_plan['visualization_tasks']:
                plot_data = self.visualization_engine.create_plot(
                    datasets, viz_task['type'], viz_task.get('parameters', {})
                )
                results[viz_task['name']] = plot_data
        
        # Format output according to requirements
        output_format = analysis_plan.get('output_format', 'object')
        formatted_result = self._format_output(results, output_format, analysis_plan)
        
        logger.info("Analysis completed successfully")
        return formatted_result
    
    def _format_output(self, results: Dict[str, Any], format_type: str, analysis_plan: Dict[str, Any]) -> Any:
        """Format the output according to specified format"""
        if format_type == 'array':
            # Convert results to array format
            if analysis_plan.get('array_fields'):
                return [results.get(field) for field in analysis_plan['array_fields']]
            else:
                return list(results.values())
        elif format_type == 'object':
            return results
        else:
            return results

# Global agent instance
agent = DataAnalystAgent()

@app.post("/api/analyze")
async def analyze_data(files: List[UploadFile] = File(...)):
    """
    Main API endpoint for data analysis
    Expects files including questions.txt and optional data files
    """
    logger.info(f"Received analysis request with {len(files)} files")
    
    try:
        # Extract files
        questions_content = None
        data_files = {}
        
        for file in files:
            content = await file.read()
            
            if file.filename == "questions.txt":
                questions_content = content.decode('utf-8')
            else:
                data_files[file.filename] = content
        
        if questions_content is None:
            raise HTTPException(status_code=400, detail="questions.txt file is required")
        
        logger.info(f"Questions content preview: {questions_content[:200]}...")
        logger.info(f"Additional files: {list(data_files.keys())}")
        
        # Process the request
        result = await agent.process_request(questions_content, data_files)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"API error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Data Analyst Agent API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/api/analyze (POST)",
            "health": "/health (GET)"
        },
        "usage": "Send POST request to /api/analyze with questions.txt and optional data files"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting Data Analyst Agent API on {host}:{port}")
    uvicorn.run(app, host=host, port=port)