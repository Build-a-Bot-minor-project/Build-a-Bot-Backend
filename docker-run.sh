#!/bin/bash

# BuildABot Docker Management Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if .env file exists
if [ ! -f .env ]; then
    print_warning ".env file not found!"
    print_status "Creating .env from env.example..."
    cp env.example .env
    print_warning "Please update your .env file with your API keys before running Docker!"
    exit 1
fi

# Parse command line arguments
case "${1:-help}" in
    "build")
        print_status "Building BuildABot Docker image..."
        docker-compose build
        print_success "Docker image built successfully!"
        ;;
    
    "up"|"start")
        print_status "Starting BuildABot with external services..."
        docker-compose up -d
        print_success "BuildABot is running!"
        print_status "Application: http://localhost:8000"
        print_status "Web Interface: http://localhost:8000/static/index.html"
        print_status "API Docs: http://localhost:8000/docs"
        ;;
    
    "down"|"stop")
        print_status "Stopping BuildABot..."
        docker-compose down
        print_success "BuildABot stopped!"
        ;;
    
    "restart")
        print_status "Restarting BuildABot..."
        docker-compose down
        docker-compose up -d
        print_success "BuildABot restarted!"
        ;;
    
    "logs")
        print_status "Showing BuildABot logs..."
        docker-compose logs -f buildabot
        ;;
    
    "status")
        print_status "BuildABot container status:"
        docker-compose ps
        ;;
    
    "shell")
        print_status "Opening shell in BuildABot container..."
        docker-compose exec buildabot /bin/bash
        ;;
    
    "clean")
        print_status "Cleaning up Docker resources..."
        docker-compose down -v
        docker system prune -f
        print_success "Docker cleanup completed!"
        ;;
    
    "help"|*)
        echo "BuildABot Docker Management Script"
        echo ""
        echo "Usage: ./docker-run.sh [command]"
        echo ""
        echo "Commands:"
        echo "  build     - Build the Docker image"
        echo "  up/start  - Start the application"
        echo "  down/stop - Stop the application"
        echo "  restart   - Restart the application"
        echo "  logs      - View application logs"
        echo "  status    - Show container status"
        echo "  shell     - Open shell in container"
        echo "  clean     - Clean up Docker resources"
        echo "  help      - Show this help message"
        echo ""
        echo "Prerequisites:"
        echo "  1. Copy env.example to .env"
        echo "  2. Update .env with your API keys"
        echo "  3. Run: ./docker-run.sh build"
        echo "  4. Run: ./docker-run.sh up"
        ;;
esac
