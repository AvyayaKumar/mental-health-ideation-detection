# Archive Old Project Directories

## ⚠️ Manual Step Required

I've created your new consolidated project, but you need to manually move the old directories to the archive folder.

## 📦 What to Archive

Move these 3 directories to `ID Archive/` folder:

1. `Ideation-Detection` (8.4GB) - Old mixed research/Flask project
2. `mental-health-ideation-detection` (1.2GB) - Old FastAPI attempt
3. `suicide-detection-workflow` (604KB) - Redundant copy

## 🔧 How to Archive (Choose One Method)

### Method 1: Using Finder (Easiest)

1. Open Finder
2. Navigate to your Desktop
3. Find the folder named `ID Archive`
4. Drag and drop these 3 folders into `ID Archive`:
   - Ideation-Detection
   - mental-health-ideation-detection
   - suicide-detection-workflow

### Method 2: Using Terminal

```bash
cd ~/Desktop

# Move old projects to archive
mv Ideation-Detection "ID Archive/"
mv mental-health-ideation-detection "ID Archive/"
mv suicide-detection-workflow "ID Archive/"

# Verify they're archived
ls "ID Archive/"
```

### Method 3: If Directories Don't Exist

If you don't see these directories on your Desktop, they may have already been moved or deleted. Check:

```bash
# Search for them
find ~ -name "Ideation-Detection" -type d 2>/dev/null
find ~ -name "mental-health-ideation-detection" -type d 2>/dev/null

# Check if already in archive
ls ~/Desktop/"ID Archive/"
```

## ✅ After Archiving

Your Desktop should only have:

```
~/Desktop/
├── suicide-ideation-detection/    # ← NEW consolidated project (use this!)
├── ID Archive/                     # ← OLD projects (archived)
│   ├── Ideation-Detection/
│   ├── mental-health-ideation-detection/
│   └── suicide-detection-workflow/
└── [other files]
```

## 🎯 Which Project to Use Now

**Always use**: `suicide-ideation-detection/`

This is your new clean, consolidated project with:
- ✅ All research code
- ✅ All deployment code
- ✅ Clean organization
- ✅ Complete documentation

## ⚠️ Don't Delete - Just Archive!

Keep the old projects in `ID Archive/` as backup. Don't delete them yet!

After 1-2 weeks of successfully using the new consolidated project, you can decide whether to delete the archive.

---

**Next Step**: See [CONSOLIDATION_SUMMARY.md](CONSOLIDATION_SUMMARY.md) for what to do next!
