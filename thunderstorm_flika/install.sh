#!/bin/bash

# ThunderSTORM for FLIKA - Installation Script
# ==============================================

echo "================================================"
echo "ThunderSTORM for FLIKA - Installation"
echo "================================================"
echo ""

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found."
    echo "Please install Python 3 and try again."
    exit 1
fi

echo "✓ Python 3 found"

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
echo "This may take a few minutes..."
echo ""

pip3 install --user numpy scipy scikit-image matplotlib pandas pywavelets tifffile

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠ Warning: Some dependencies may not have installed correctly."
    echo "You may need to install them manually:"
    echo "  pip3 install numpy scipy scikit-image matplotlib pandas pywavelets tifffile"
else
    echo ""
    echo "✓ Dependencies installed successfully"
fi

# Determine FLIKA plugin directory
if [ -d "$HOME/.FLIKA/plugins" ]; then
    PLUGIN_DIR="$HOME/.FLIKA/plugins"
else
    echo ""
    echo "Creating FLIKA plugin directory..."
    mkdir -p "$HOME/.FLIKA/plugins"
    PLUGIN_DIR="$HOME/.FLIKA/plugins"
fi

echo "✓ Plugin directory: $PLUGIN_DIR"

# Copy plugin files
echo ""
echo "Installing ThunderSTORM plugin..."

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Copy the entire plugin directory
if [ -d "$SCRIPT_DIR" ]; then
    cp -r "$SCRIPT_DIR" "$PLUGIN_DIR/thunderstorm_flika"
    echo "✓ Plugin files copied to $PLUGIN_DIR/thunderstorm_flika"
else
    echo "Error: Could not find plugin directory"
    exit 1
fi

# Verify installation
echo ""
echo "Verifying installation..."

if [ -f "$PLUGIN_DIR/thunderstorm_flika/__init__.py" ] && \
   [ -f "$PLUGIN_DIR/thunderstorm_flika/info.xml" ] && \
   [ -f "$PLUGIN_DIR/thunderstorm_flika/about.html" ] && \
   [ -d "$PLUGIN_DIR/thunderstorm_flika/thunderstorm_python" ]; then
    echo "✓ All required files present"
else
    echo "⚠ Warning: Some files may be missing"
fi

echo ""
echo "================================================"
echo "Installation Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Restart FLIKA"
echo "2. Look for 'ThunderSTORM' in the Plugins menu"
echo "3. Try 'Plugins → ThunderSTORM → Quick Analysis'"
echo ""
echo "For documentation, see:"
echo "  - README.md in the plugin directory"
echo "  - about.html (accessible from FLIKA Plugin Manager)"
echo ""
echo "Enjoy super-resolution imaging with ThunderSTORM!"
echo "================================================"
