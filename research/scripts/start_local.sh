#!/bin/bash
# Start the application locally with Docker Compose

echo "🚀 Starting Mental Health Ideation Detection locally with Docker..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Load environment variables if .env exists
if [ -f .env ]; then
    echo "✓ Loading environment variables from .env"
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "⚠️  No .env file found. Using default values."
    echo "   Copy .env.example to .env and configure it for production."
fi

# Build and start containers
echo "📦 Building Docker images..."
docker-compose build

echo "🎬 Starting services (Web, Redis, Celery)..."
docker-compose up -d

# Wait for services to start
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo ""
    echo "✅ Application is running!"
    echo ""
    echo "🌐 Access the app at: http://localhost:8080"
    echo "🔍 Health check: http://localhost:8080/health"
    echo "👨‍💼 Admin dashboard: http://localhost:8080/admin"
    echo ""
    echo "📋 View logs: docker-compose logs -f"
    echo "🛑 Stop services: docker-compose down"
    echo ""
else
    echo "❌ Failed to start services. Check logs with: docker-compose logs"
    exit 1
fi
