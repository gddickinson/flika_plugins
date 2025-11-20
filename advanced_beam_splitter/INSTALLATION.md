# Installation Guide - Advanced Beam Splitter Plugin

## Prerequisites

Before installing the Advanced Beam Splitter plugin, ensure you have:

1. **FLIKA Installed**
   - Download from: https://github.com/flika-org/flika
   - Or via pip: `pip install flika`
   - Minimum version: 0.2.23 or higher

2. **Python Environment**
   - Python 3.7 or higher recommended
   - 64-bit Python installation

3. **Required Dependencies**
   Most dependencies come with FLIKA, but verify you have:
   - numpy
   - scipy
   - PyQt5 or PySide2
   - pyqtgraph
   - scikit-image

## Installation Methods

### Method 1: FLIKA Plugin Manager (Recommended)

This is the easiest method if the plugin is published to the FLIKA plugin repository:

1. **Open FLIKA**
   ```
   Start FLIKA application
   ```

2. **Access Plugin Manager**
   ```
   Menu: Plugins → Plugin Manager
   ```

3. **Search and Install**
   ```
   Search: "Advanced Beam Splitter"
   Click: Download
   Click: Install
   ```

4. **Restart FLIKA**
   ```
   Close and reopen FLIKA
   Plugin will appear under Plugins menu
   ```

### Method 2: Manual Installation

If installing from downloaded files:

#### Step 1: Locate FLIKA Plugins Directory

**Windows:**
```
C:\Users\<YourUsername>\AppData\Roaming\flika\plugins\
```

**macOS:**
```
~/Library/Application Support/flika/plugins/
```

**Linux:**
```
~/.local/share/flika/plugins/
```

To find it programmatically in FLIKA:
```python
import flika
print(flika.get_plugins_directory())
```

#### Step 2: Create Plugin Folder

Create a new folder named `advanced_beam_splitter` in the plugins directory:

```
<plugins_directory>/advanced_beam_splitter/
```

#### Step 3: Copy Files

Copy these files into the `advanced_beam_splitter` folder:

```
advanced_beam_splitter/
├── __init__.py
├── advanced_beam_splitter.py
├── info.xml
└── README.md  (optional, for reference)
```

#### Step 4: Verify File Structure

Your directory should look like:
```
<plugins_directory>/
└── advanced_beam_splitter/
    ├── __init__.py
    ├── advanced_beam_splitter.py
    ├── info.xml
    └── README.md
```

#### Step 5: Restart FLIKA

1. Close FLIKA completely
2. Reopen FLIKA
3. Check **Plugins** menu for "Advanced Beam Splitter"

### Method 3: Development Installation

For developers who want to modify the plugin:

1. **Clone or Download Source**
   ```bash
   git clone <repository_url>
   cd advanced_beam_splitter
   ```

2. **Create Symbolic Link** (Recommended for development)
   
   **Windows (PowerShell as Administrator):**
   ```powershell
   New-Item -ItemType SymbolicLink -Path "C:\Users\<YourUsername>\AppData\Roaming\flika\plugins\advanced_beam_splitter" -Target "<path_to_source>\advanced_beam_splitter"
   ```
   
   **macOS/Linux:**
   ```bash
   ln -s /path/to/source/advanced_beam_splitter ~/Library/Application Support/flika/plugins/advanced_beam_splitter
   ```

3. **Install Dependencies** (if missing)
   ```bash
   pip install numpy scipy pyqtgraph scikit-image
   ```

4. **Restart FLIKA**

## Verifying Installation

### Test 1: Menu Check
1. Open FLIKA
2. Click **Plugins** in menu bar
3. Look for "Advanced Beam Splitter" in the list

### Test 2: Programmatic Check
In FLIKA's script editor or Python console:
```python
from plugins.advanced_beam_splitter import advanced_beam_splitter
print("Plugin loaded successfully!")
print(advanced_beam_splitter.__doc__)
```

### Test 3: Launch Plugin
1. Click **Plugins → Advanced Beam Splitter**
2. GUI window should open with all controls visible

## Troubleshooting Installation

### Issue 1: Plugin doesn't appear in menu

**Possible causes:**
- Files not in correct directory
- Missing `__init__.py` file
- Python import errors

**Solutions:**
1. Verify file locations:
   ```python
   import flika
   print(flika.get_plugins_directory())
   ```

2. Check FLIKA console for error messages on startup

3. Manually test import:
   ```python
   import sys
   sys.path.append('/path/to/plugins/directory')
   from advanced_beam_splitter import advanced_beam_splitter
   ```

### Issue 2: Import errors or missing dependencies

**Error message like:**
```
ImportError: No module named 'scipy'
```

**Solution:**
Install missing package:
```bash
pip install scipy
# or
conda install scipy
```

Common missing packages:
```bash
pip install numpy scipy pyqtgraph scikit-image
```

### Issue 3: FLIKA can't find plugin directory

**Solution:**
Create the directory manually:

**Windows:**
```powershell
New-Item -ItemType Directory -Force -Path "$env:APPDATA\flika\plugins"
```

**macOS/Linux:**
```bash
mkdir -p ~/Library/Application\ Support/flika/plugins
```

### Issue 4: Permission errors

**Windows:**
- Run FLIKA as Administrator
- Check folder permissions

**macOS/Linux:**
- Ensure you have write permissions:
```bash
chmod -R u+w ~/Library/Application\ Support/flika/plugins
```

### Issue 5: Plugin loads but crashes

**Possible causes:**
- Incompatible FLIKA version
- Missing dependencies
- Corrupted files

**Solutions:**

1. Check FLIKA version:
   ```python
   import flika
   print(flika.__version__)
   ```
   Should be 0.2.23 or higher

2. Check for errors in FLIKA console

3. Reinstall dependencies:
   ```bash
   pip install --upgrade numpy scipy pyqtgraph scikit-image
   ```

4. Re-download plugin files (may be corrupted)

## Updating the Plugin

### Method 1: Via Plugin Manager
```
Plugins → Plugin Manager → Advanced Beam Splitter → Update
```

### Method 2: Manual Update
1. Delete old plugin folder
2. Download new version
3. Follow installation steps above
4. Restart FLIKA

### Method 3: Development Update
If using symbolic link:
```bash
cd /path/to/source
git pull origin main
# FLIKA will use updated files automatically
```

## Uninstalling

### Complete Removal
Delete the plugin folder:

**Windows:**
```powershell
Remove-Item -Recurse -Force "$env:APPDATA\flika\plugins\advanced_beam_splitter"
```

**macOS/Linux:**
```bash
rm -rf ~/Library/Application\ Support/flika/plugins/advanced_beam_splitter
```

Then restart FLIKA.

### Temporary Disable
Rename the folder:
```
advanced_beam_splitter → _advanced_beam_splitter_disabled
```

## Platform-Specific Notes

### Windows
- Use PowerShell for symbolic links (requires admin)
- Path separators: use `\` or raw strings `r"path\to\file"`
- May need to add Python to PATH

### macOS
- May need to allow FLIKA in Security & Privacy settings
- Use Terminal for installation commands
- Application Support folder may be hidden

### Linux
- Usually most straightforward installation
- Ensure Python 3 is default (not Python 2)
- Check file permissions

## Environment-Specific Installation

### Anaconda/Miniconda
```bash
conda activate flika_env
conda install -c conda-forge numpy scipy pyqtgraph scikit-image
# Then install plugin as above
```

### Virtual Environment
```bash
python -m venv flika_venv
source flika_venv/bin/activate  # Linux/macOS
# or
flika_venv\Scripts\activate  # Windows

pip install flika numpy scipy pyqtgraph scikit-image
# Then install plugin as above
```

## Post-Installation Setup

### Recommended: Create Test Dataset

1. **Download test images:**
   - TetraSpeck bead images
   - Or use your own dual-channel TIRF data

2. **Test basic functionality:**
   - Load two channel images
   - Launch plugin
   - Try auto-alignment
   - Test preview

3. **Test advanced features:**
   - Background subtraction
   - Photobleaching correction
   - Save results

### Configuration

The plugin uses FLIKA's configuration. To set preferences:

```python
import flika.global_vars as g
# Settings are saved automatically
```

## Getting Help

If installation issues persist:

1. **Check FLIKA Documentation**
   - https://flika-org.github.io/

2. **FLIKA Forums**
   - Search for similar issues
   - Post detailed error messages

3. **Check Console Output**
   - Look for error messages when FLIKA starts
   - Screenshot any error dialogs

4. **Provide System Information**
   ```python
   import sys
   import flika
   print(f"Python: {sys.version}")
   print(f"FLIKA: {flika.__version__}")
   print(f"Platform: {sys.platform}")
   ```

## Success Checklist

✅ FLIKA version 0.2.23 or higher installed  
✅ All dependencies (numpy, scipy, pyqtgraph, scikit-image) installed  
✅ Plugin files in correct directory structure  
✅ Plugin appears in Plugins menu  
✅ Plugin GUI launches without errors  
✅ Can load and preview test images  

If all checkboxes are complete, you're ready to use the Advanced Beam Splitter!

---

**Next Steps:** See [README.md](README.md) for usage instructions and [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for keyboard shortcuts.
