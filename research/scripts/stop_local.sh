#!/bin/bash
# Stop local Docker services

echo "🛑 Stopping Mental Health Ideation Detection services..."

docker-compose down

echo "✅ All services stopped!"
echo ""
echo "💾 Data is preserved in Docker volumes."
echo "   To remove volumes as well, run: docker-compose down -v"
