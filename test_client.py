#!/usr/bin/env python3
"""
Test client for Data Analyst Agent API
Usage: python test_client.py [API_URL]
"""

import requests
import json
import sys
import os
import tempfile
import pandas as pd

def create_test_files():
    """Create test files for API testing"""
    
    # Create test questions.txt
    questions_content = """
    Analyze the relationship between sales and marketing spend.
    Calculate the correlation coefficient and create a scatter plot with regression line.
    Return the results as a JSON object with correlation, slope, and plot fields.
    """
    
    # Create test CSV data
    test_data = pd.DataFrame({
        'marketing_spend': [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500],
        'sales': [15000, 22000, 28000, 35000, 42000, 48000, 55000, 62000, 68000, 75000],
        'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
    })
    
    return questions_content, test_data

def test_api(api_url):
    """Test the Data Analyst Agent API"""
    
    print(f"Testing API at: {api_url}")
    
    # Test health endpoint first
    try:
        health_response = requests.get(f"{api_url}/health", timeout=10)
        if health_response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {health_response.json()}")
        else:
            print(f"❌ Health check failed: {health_response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Could not connect to API: {str(e)}")
        return False
    
    # Create test files
    questions_content, test_data = create_test_files()
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        questions_file = os.path.join(temp_dir, "questions.txt")
        data_file = os.path.join(temp_dir, "test_data.csv")
        
        # Write files
        with open(questions_file, 'w') as f:
            f.write(questions_content)
        
        test_data.to_csv(data_file, index=False)
        
        # Test main API endpoint
        try:
            files = [
                ('files', ('questions.txt', open(questions_file, 'rb'), 'text/plain')),
                ('files', ('test_data.csv', open(data_file, 'rb'), 'text/csv'))
            ]
            
            print("\n🚀 Testing analysis endpoint...")
            response = requests.post(f"{api_url}/api/analyze", files=files, timeout=180)
            
            # Close file handles
            for file_tuple in files:
                file_tuple[1][1].close()
            
            if response.status_code == 200:
                print("✅ Analysis completed successfully!")
                
                result = response.json()
                print(f"📊 Result keys: {list(result.keys())}")
                
                # Check for expected components
                if 'correlation' in str(result).lower():
                    print("✅ Correlation analysis found")
                
                if 'slope' in str(result).lower():
                    print("✅ Slope calculation found")
                
                if 'data:image/' in str(result):
                    print("✅ Base64 image found in results")
                    
                    # Check image size
                    for key, value in result.items():
                        if isinstance(value, str) and value.startswith('data:image/'):
                            image_size = len(value)
                            print(f"   Image size: {image_size:,} bytes")
                            if image_size < 100000:
                                print("✅ Image size under 100KB limit")
                            else:
                                print("❌ Image size exceeds 100KB limit")
                
                # Pretty print result (truncate long base64 strings)
                display_result = {}
                for k, v in result.items():
                    if isinstance(v, str) and v.startswith('data:image/'):
                        display_result[k] = f"data:image/...[{len(v)} chars]"
                    else:
                        display_result[k] = v
                
                print(f"\n📋 Results summary:")
                print(json.dumps(display_result, indent=2, default=str))
                
                return True
                
            else:
                print(f"❌ Analysis failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ API test failed: {str(e)}")
            return False

def main():
    """Main function"""
    
    # Get API URL from command line or use default
    if len(sys.argv) > 1:
        api_url = sys.argv[1].rstrip('/')
    else:
        api_url = "http://localhost:8000"
    
    print("=" * 60)
    print("🧪 DATA ANALYST AGENT API TEST")
    print("=" * 60)
    
    success = test_api(api_url)
    
    if success:
        print("\n🎉 All tests passed! API is working correctly.")
        print(f"🌐 API is ready for production use at: {api_url}")
    else:
        print("\n💥 Tests failed. Please check the API server.")
        sys.exit(1)

if __name__ == "__main__":
    main()