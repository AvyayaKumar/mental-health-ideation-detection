#!/bin/bash
# STEP 1: Push to GitHub
# Run this script to automatically commit and push all changes

set -e

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║               STEP 1: PUSHING TO GITHUB                       ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Verify we're in the right directory
if [ ! -f "apps/app.py" ]; then
    echo "❌ Error: Run this from project root"
    exit 1
fi

echo "📦 Adding all files to git..."
git add .

echo ""
echo "📝 Creating commit..."
git commit -m "Railway deployment: Docker + Celery + Redis configuration

- Add Dockerfile and docker-compose.yml
- Configure Railway with railway.toml and Procfile
- Add Celery worker for background tasks
- Configure Redis for message queue and caching
- Add production dependencies (gunicorn, celery, redis)
- Include comprehensive deployment documentation
- Ready for production deployment to mentalhealthideation.com"

echo ""
echo "🚀 Pushing to GitHub..."
git push origin main

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                    ✅ STEP 1 COMPLETE!                         ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "✅ Code pushed to GitHub successfully!"
echo ""
echo "📋 NEXT STEP: Deploy to Railway"
echo ""
echo "Go to: https://railway.app"
echo ""
echo "Then follow STEP 2 in DEPLOY_NOW.txt"
echo "Or run: cat DEPLOY_NOW.txt | grep -A 20 'STEP 2'"
echo ""
