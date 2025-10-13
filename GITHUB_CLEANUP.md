# GitHub Repository Clean-Up Guide

## 📊 Current Situation

You have **2 GitHub repositories** that need consolidation:

1. **`ideation-detection-production`**
   - Old Flask-based mixed code
   - Connected to archived Ideation-Detection folder
   - Status: ❌ Should be deleted

2. **`mental-health-ideation-detection`**
   - Newer FastAPI deployment code
   - Has latest Railway configuration
   - Status: ✅ Keep and update

---

## 🎯 Recommended Action Plan

### Step 1: Push Consolidated Code to GitHub (10 minutes)

```bash
cd ~/Desktop/suicide-ideation-detection

# Initialize git
git init

# Connect to your better GitHub repo
git remote add origin https://github.com/AvyayaKumar/mental-health-ideation-detection.git

# Add all files
git add .

# Create comprehensive first commit
git commit -m "Consolidate: Unified research and deployment project

Features:
- Separated research code (training, experiments) into research/
- Separated deployment code (FastAPI, Railway) into deployment/
- Added comprehensive documentation (10+ guides)
- Clean project structure
- Industry-grade deployment: Docker + Railway + Celery + Redis
- Complete research pipeline for publication

Organization:
- research/: All ML training code, configs, models
- deployment/: Production FastAPI application
- docs/: Research plans and publication materials
- Comprehensive READMEs throughout"

# Push to GitHub (replaces old code)
git branch -M main
git push -u origin main --force
```

**Note**: The `--force` flag will replace the old code in the repo. This is intentional!

---

### Step 2: Delete Old Repository on GitHub (5 minutes)

#### Delete: `ideation-detection-production`

1. **Go to**: https://github.com/AvyayaKumar/ideation-detection-production

2. **Click Settings** (top navigation bar)

3. **Scroll to bottom** → Find "Danger Zone"

4. **Click**: "Delete this repository"

5. **Type to confirm**: `AvyayaKumar/ideation-detection-production`

6. **Click**: "I understand the consequences, delete this repository"

**Done!** The old repo is deleted.

---

### Step 3: Update Repository Name (Optional, 5 minutes)

If you want to rename `mental-health-ideation-detection` to something clearer:

1. **Go to**: https://github.com/AvyayaKumar/mental-health-ideation-detection

2. **Click Settings**

3. **Repository name** → Change to: `suicide-ideation-detection`

4. **Click "Rename"**

5. **Update local remote** (in your terminal):
   ```bash
   cd ~/Desktop/suicide-ideation-detection
   git remote set-url origin https://github.com/AvyayaKumar/suicide-ideation-detection.git
   ```

---

## 🔄 Alternative: Create Brand New Repo

If you prefer a completely fresh start:

### Option B: New Repository

1. **Create new repo on GitHub**:
   - Go to: https://github.com/new
   - Name: `suicide-ideation-detection`
   - Description: "ML research & production system for suicide ideation detection"
   - Public or Private (your choice)
   - **Don't** initialize with README (we have our own)
   - Click "Create repository"

2. **Push consolidated code**:
   ```bash
   cd ~/Desktop/suicide-ideation-detection
   git init
   git remote add origin https://github.com/AvyayaKumar/suicide-ideation-detection.git
   git add .
   git commit -m "Initial commit: Consolidated research and deployment project"
   git branch -M main
   git push -u origin main
   ```

3. **Delete both old repos**:
   - Delete `ideation-detection-production`
   - Delete `mental-health-ideation-detection`
   - (Follow deletion steps from Step 2 above)

---

## ✅ Final Result

After clean-up, you'll have:

**GitHub**:
- ✅ 1 repository: `mental-health-ideation-detection` (or renamed to `suicide-ideation-detection`)
- ❌ Deleted: `ideation-detection-production`

**Local**:
- ✅ 1 project: `~/Desktop/suicide-ideation-detection/`
- 📦 Archived: Old projects in `ID Archive/`

**Connected**:
```bash
# Your consolidated project points to GitHub
cd ~/Desktop/suicide-ideation-detection
git remote -v
# Should show: mental-health-ideation-detection (or your renamed repo)
```

---

## 🔒 Before You Delete - Checklist

Make sure:

- [ ] Consolidated project is pushed to GitHub
- [ ] Check GitHub repo has all your code
- [ ] Verify research code is there: `research/src/`
- [ ] Verify deployment code is there: `deployment/backend/`
- [ ] Old projects are archived in `ID Archive/`

Only delete GitHub repos after confirming everything is pushed!

---

## 🆘 Troubleshooting

### "Push rejected - not fast-forward"

Use `--force`:
```bash
git push -u origin main --force
```

This is safe because you're replacing old messy code with clean consolidated code.

### "Authentication failed"

Setup GitHub authentication:
```bash
# Use GitHub CLI
brew install gh
gh auth login

# Or use SSH keys (recommended)
# See: https://docs.github.com/en/authentication/connecting-to-github-with-ssh
```

### "Want to keep commit history?"

If you want to preserve old commit history:
```bash
# Don't use --force
# Instead, pull first, then push
git pull origin main --allow-unrelated-histories
git push origin main
```

---

## 📞 Summary

**Recommended Steps**:
1. Push consolidated code to `mental-health-ideation-detection`
2. Delete `ideation-detection-production` on GitHub
3. Optionally rename repo to `suicide-ideation-detection`

**Time Required**: 15-20 minutes

**Result**: Clean GitHub with 1 unified repository!

---

**Ready?** Start with Step 1 above!
