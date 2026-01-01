# Troubleshooting Visualization Issues on Google Colab

## Problem
When running `python tools/visualize_results.py` on Colab, you get:
```
No .npz results found under results/ - run experiments first or point to another directory.
```

## Solution

### Method 1: Use the Wrapper Script (Recommended)
Instead of:
```bash
!python tools/visualize_results.py
```

Use:
```bash
!python visualize.py
```

The wrapper script automatically handles path issues and is Colab-compatible.

### Method 2: Change Working Directory First
```bash
import os
os.chdir('/content/drive/MyDrive/sr-fedopt')
!python tools/visualize_results.py
```

### Method 3: Verify Results Exist First
```bash
# Check what files exist
!find results -name "*.npz" | wc -l

# Should show a number > 0
# If 0, experiments haven't completed yet
```

## Updated Colab Cell

Replace this cell in your notebook:

```python
# Generate all plots and CSV files
!python visualize.py
```

(Previously was: `!python tools/visualize_results.py`)

## Why This Happens

1. **Working Directory:** Colab may change directories, making relative paths fail
2. **Path Resolution:** The wrapper script automatically finds the project root
3. **Import Issues:** Direct import from tools/ can fail if working directory is wrong

## Files Generated

After running `visualize.py`, you'll get:
- `results/plots/` - Contains all PNG plots
- `results/plots/*.csv` - Contains accuracy curves and statistics
- `results/plots/*_statistics_summary.csv` - Summary statistics

## Next Steps

1. Update your Colab notebook with `!python visualize.py`
2. Run the cell
3. View plots with the display cells below

---

**Updated:** January 2026
