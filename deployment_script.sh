#!/bin/bash

# Data Analyst Agent API Deployment Script
echo "Starting Data Analyst Agent API deployment..."

# Check if Python 3.8+ is installed
python_major=$(python3 -c "import sys; print(sys.version_info[0])")
python_minor=$(python3 -c "import sys; print(sys.version_info[1])")
if [ "$python_major" -lt 3 ] || { [ "$python_major" -eq 3 ] && [ "$python_minor" -lt 8 ]; }; then
    echo "Error: Python 3.8+ required. Current version: ${python_major}.${python_minor}"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Set environment variables
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-8000}

echo "Starting API server on $HOST:$PORT"

# Start the API server
python main.py &
SERVER_PID=$!

echo "Server started with PID: $SERVER_PID"
echo "API is available at: http://$HOST:$PORT"
echo "Health check: http://$HOST:$PORT/health"
echo "Main endpoint: http://$HOST:$PORT/api/analyze"

# Optional: Set up ngrok for public access (if ngrok is installed)
if command -v ngrok &> /dev/null; then
    echo "Setting up ngrok tunnel..."
    ngrok http $PORT --log=stdout > ngrok.log 2>&1 &
    NGROK_PID=$!
    
    # Wait a moment for ngrok to start
    sleep 3
    
    # Extract public URL
    PUBLIC_URL=$(curl -s localhost:4040/api/tunnels | python3 -c "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])" 2>/dev/null)
    
    if [ ! -z "$PUBLIC_URL" ]; then
        echo "Public URL: $PUBLIC_URL"
        echo "Public API endpoint: $PUBLIC_URL/api/analyze"
    fi
    
    # Save PIDs for cleanup
    echo $SERVER_PID > server.pid
    echo $NGROK_PID > ngrok.pid
else
    echo "ngrok not found. Server is only accessible locally."
    echo $SERVER_PID > server.pid
fi

echo "Deployment complete!"
echo ""
echo "To test the API, you can use:"
echo "curl -X POST -F 'files=@questions.txt' -F 'files=@data.csv' http://$HOST:$PORT/api/analyze"
echo ""
echo "To stop the server, run: kill \$(cat server.pid)"
if [ -f "ngrok.pid" ]; then
    echo "To stop ngrok, run: kill \$(cat ngrok.pid)"
fi