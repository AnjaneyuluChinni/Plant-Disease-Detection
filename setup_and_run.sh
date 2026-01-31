#!/bin/bash
# Startup script for Plant Disease Detection
# Usage: bash setup_and_run.sh

set -e  # Exit on error

echo "========================================"
echo "Plant Disease Detection - Setup & Run"
echo "========================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python: $python_version"

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Activated${NC}"

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create directories
echo -e "\n${YELLOW}Creating project directories...${NC}"
mkdir -p datasets/raw
mkdir -p datasets/yolo_format
mkdir -p models
mkdir -p backend/uploads
mkdir -p frontend/templates
mkdir -p frontend/static
mkdir -p notebooks
echo -e "${GREEN}✓ Directories created${NC}"

# Check dataset
echo -e "\n${YELLOW}Checking dataset...${NC}"
if [ ! -d "datasets/raw" ] || [ -z "$(ls -A datasets/raw)" ]; then
    echo -e "${RED}⚠ PlantVillage dataset not found${NC}"
    echo "Download from: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset"
    echo "Extract to: datasets/raw/"
    echo ""
    echo "Then run: python utils/dataset_converter.py"
else
    echo -e "${GREEN}✓ Dataset found${NC}"
    
    # Convert dataset
    echo -e "\n${YELLOW}Converting dataset to YOLO format...${NC}"
    python utils/dataset_converter.py
fi

# Check model
echo -e "\n${YELLOW}Checking model...${NC}"
if [ ! -f "models/best.pt" ]; then
    echo -e "${RED}⚠ Model not found${NC}"
    echo "Options:"
    echo "1. Download pre-trained: wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt -O models/best.pt"
    echo "2. Train your own: python utils/train_yolov5.py"
else
    echo -e "${GREEN}✓ Model found${NC}"
fi

# Start Flask server
echo -e "\n${GREEN}========================================"
echo "Starting Flask server..."
echo "========================================${NC}"
echo -e "\nServer will be available at: ${GREEN}http://localhost:5000${NC}"
echo "Press Ctrl+C to stop"
echo ""

cd backend
python app.py
