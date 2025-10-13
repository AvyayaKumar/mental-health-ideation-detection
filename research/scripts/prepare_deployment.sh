#!/bin/bash
# Automated Deployment Preparation Script
# This script does all the automated preparation work

set -e  # Exit on any error

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     RAILWAY DEPLOYMENT PREPARATION - AUTOMATED SETUP          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "apps/app.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "   cd /Users/avyayakumar/Desktop/Ideation-Detection"
    exit 1
fi

echo "📍 Current directory: $(pwd)"
echo ""

# Step 1: Generate SECRET_KEY
echo -e "${BLUE}[1/6] Generating SECRET_KEY...${NC}"
SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
echo -e "${GREEN}✓ SECRET_KEY generated${NC}"
echo ""

# Step 2: Create .env file for reference
echo -e "${BLUE}[2/6] Creating .env file (for your reference)...${NC}"
cat > .env << EOF
# Generated on $(date)
# THIS FILE IS FOR REFERENCE ONLY - DO NOT COMMIT TO GIT

# COPY THESE TO RAILWAY ENVIRONMENT VARIABLES

SECRET_KEY=${SECRET_KEY}
MODEL_PATH=/app/results/distilbert-seed42/final_model
EMAIL_NOTIFICATIONS_ENABLED=false
ADMIN_EMAIL=avyaya.kumar@gmail.com
PYTHONUNBUFFERED=1

# Optional: Email Notifications (configure later)
# EMAIL_NOTIFICATIONS_ENABLED=true
# SMTP_HOST=smtp.gmail.com
# SMTP_PORT=587
# SMTP_USER=avyaya.kumar@gmail.com
# SMTP_PASSWORD=your-gmail-app-password-here
EOF
echo -e "${GREEN}✓ .env file created${NC}"
echo ""

# Step 3: Verify model files exist
echo -e "${BLUE}[3/6] Verifying model files...${NC}"
MODEL_DIR="results/distilbert-seed42/final_model"
if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
    echo "❌ Error: Model file not found at $MODEL_DIR/model.safetensors"
    exit 1
fi
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "❌ Error: Model config not found at $MODEL_DIR/config.json"
    exit 1
fi
if [ ! -f "$MODEL_DIR/tokenizer.json" ]; then
    echo "❌ Error: Tokenizer not found at $MODEL_DIR/tokenizer.json"
    exit 1
fi
echo -e "${GREEN}✓ All model files verified${NC}"
echo ""

# Step 4: Check git status
echo -e "${BLUE}[4/6] Checking git status...${NC}"
git status --short
echo -e "${GREEN}✓ Git status checked${NC}"
echo ""

# Step 5: Verify Docker configuration files exist
echo -e "${BLUE}[5/6] Verifying deployment configuration files...${NC}"
required_files=(
    "Dockerfile"
    "docker-compose.yml"
    ".dockerignore"
    "railway.toml"
    "railway.json"
    "Procfile"
    ".env.example"
    "apps/worker.py"
    "apps/tasks.py"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing: $file"
        exit 1
    fi
done
echo -e "${GREEN}✓ All configuration files present${NC}"
echo ""

# Step 6: Create deployment instructions file
echo -e "${BLUE}[6/6] Creating deployment instructions...${NC}"
cat > DEPLOY_NOW.txt << EOF
╔═══════════════════════════════════════════════════════════════════════════╗
║                   READY TO DEPLOY - FOLLOW THESE STEPS                    ║
╚═══════════════════════════════════════════════════════════════════════════╝

Your SECRET_KEY: ${SECRET_KEY}

🎯 NEXT STEPS - DO THESE NOW:

═══════════════════════════════════════════════════════════════════════════
STEP 1: PUSH TO GITHUB (2 minutes)
═══════════════════════════════════════════════════════════════════════════

Run these commands in your terminal:

cd /Users/avyayakumar/Desktop/Ideation-Detection
git add .
git commit -m "Railway deployment: Docker + Celery + Redis configuration"
git push origin main

═══════════════════════════════════════════════════════════════════════════
STEP 2: CREATE RAILWAY PROJECT (3 minutes)
═══════════════════════════════════════════════════════════════════════════

1. Open your browser: https://railway.app
2. Click "Login" (use GitHub)
3. Click "New Project"
4. Select "Deploy from GitHub repo"
5. Choose: kianjavaheri/Ideation-Detection
6. Click "Deploy Now"
7. Wait 5 minutes for deployment

═══════════════════════════════════════════════════════════════════════════
STEP 3: ADD REDIS SERVICE (1 minute)
═══════════════════════════════════════════════════════════════════════════

In your Railway project:
1. Click "+ New"
2. Click "Database"
3. Select "Redis"
4. Wait 30 seconds

═══════════════════════════════════════════════════════════════════════════
STEP 4: SET ENVIRONMENT VARIABLES (3 minutes)
═══════════════════════════════════════════════════════════════════════════

In Railway → Click "Web Service" → Go to "Variables" tab

Click "+ New Variable" and add EACH of these:

Variable Name: SECRET_KEY
Value: ${SECRET_KEY}

Variable Name: MODEL_PATH
Value: /app/results/distilbert-seed42/final_model

Variable Name: EMAIL_NOTIFICATIONS_ENABLED
Value: false

Variable Name: ADMIN_EMAIL
Value: avyaya.kumar@gmail.com

Variable Name: PYTHONUNBUFFERED
Value: 1

Then click "Deploy" button (will restart app)

═══════════════════════════════════════════════════════════════════════════
STEP 5: TEST RAILWAY DEPLOYMENT (2 minutes)
═══════════════════════════════════════════════════════════════════════════

1. In Railway, click on "Web Service"
2. Click "Settings" → "Domains"
3. Copy the Railway URL (looks like: xxx.up.railway.app)
4. Open that URL in your browser
5. Try analyzing some sample text
6. Verify it works!

Test URL: https://[your-app].up.railway.app/health
Should return: {"status": "healthy"}

═══════════════════════════════════════════════════════════════════════════
STEP 6: ADD CUSTOM DOMAIN (5 minutes)
═══════════════════════════════════════════════════════════════════════════

A. In Railway:
   1. Web Service → Settings → Domains
   2. Click "+ Custom Domain"
   3. Enter: mentalhealthideation.com
   4. Railway will show DNS records to add

B. At your domain registrar (GoDaddy/Namecheap/etc):

   Add CNAME Record #1:
   Type: CNAME
   Name: @  (or blank)
   Value: [your-app].up.railway.app  (Railway will show this)
   TTL: 3600

   Add CNAME Record #2:
   Type: CNAME
   Name: www
   Value: [your-app].up.railway.app  (same as above)
   TTL: 3600

C. Wait 10-30 minutes for DNS propagation

═══════════════════════════════════════════════════════════════════════════
STEP 7: VERIFY PRODUCTION (2 minutes)
═══════════════════════════════════════════════════════════════════════════

After DNS propagates, test:

1. https://mentalhealthideation.com
2. https://www.mentalhealthideation.com
3. https://mentalhealthideation.com/health
4. https://mentalhealthideation.com/admin

All should work with HTTPS (🔒 lock icon)!

═══════════════════════════════════════════════════════════════════════════
OPTIONAL: ADD CELERY WORKER (5 minutes)
═══════════════════════════════════════════════════════════════════════════

For background email notifications:

1. Railway → "+ New" → "Empty Service"
2. Name it: celery-worker
3. Settings → Source → Connect same GitHub repo
4. Settings → Deploy → Start Command:
   celery -A apps.worker.celery_app worker --loglevel=info --concurrency=2
5. Variables → "Copy All From" → Select "Web Service"
6. Click "Deploy"

═══════════════════════════════════════════════════════════════════════════
OPTIONAL: ENABLE EMAIL NOTIFICATIONS (5 minutes)
═══════════════════════════════════════════════════════════════════════════

1. Get Gmail App Password:
   - Go to: https://myaccount.google.com/apppasswords
   - Create password for "Mental Health Ideation Detection"
   - Copy the 16-character password

2. Update Railway Variables (Web Service):
   EMAIL_NOTIFICATIONS_ENABLED=true
   SMTP_USER=avyaya.kumar@gmail.com
   SMTP_PASSWORD=[paste-16-char-password]
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587

3. If you added Celery Worker, update its variables too

═══════════════════════════════════════════════════════════════════════════

🎉 DONE! Your app is live at: https://mentalhealthideation.com

Cost: ~\$10-15/month (Railway gives \$5 free credit first month!)

═══════════════════════════════════════════════════════════════════════════
EOF

echo -e "${GREEN}✓ Instructions created: DEPLOY_NOW.txt${NC}"
echo ""

# Summary
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                  ✅ PREPARATION COMPLETE!                      ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}Everything is ready for deployment!${NC}"
echo ""
echo -e "${YELLOW}📋 YOUR NEXT STEPS:${NC}"
echo ""
echo "1. Read the file: DEPLOY_NOW.txt"
echo "   cat DEPLOY_NOW.txt"
echo ""
echo "2. Start with STEP 1 (Push to GitHub)"
echo ""
echo -e "${BLUE}Your SECRET_KEY is saved in .env file${NC}"
echo -e "${YELLOW}⚠️  DO NOT commit .env to Git (already in .gitignore)${NC}"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""
