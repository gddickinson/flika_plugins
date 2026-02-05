# tracking_results_plotter/setup.py
"""
Setup and installation script for the Tracking Results Plotter FLIKA plugin.

This script handles plugin installation, dependency checking, configuration setup,
and provides utilities for plugin management and updates.
"""

import os
import sys
import shutil
import json
import subprocess
import platform
import importlib.util
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import tempfile
import zipfile
import urllib.request
import urllib.error

# Version information
PLUGIN_NAME = "tracking_results_plotter"
PLUGIN_VERSION = "1.0.0"
FLIKA_MIN_VERSION = "0.2.25"

# Required dependencies
REQUIRED_DEPENDENCIES = {
    'numpy': '>=1.19.0',
    'pandas': '>=1.3.0',
    'qtpy': '>=1.9.0'  # Should come with FLIKA
}

OPTIONAL_DEPENDENCIES = {
    'matplotlib': '>=3.3.0',
    'scipy': '>=1.7.0',
    'seaborn': '>=0.11.0'
}

class PluginInstaller:
    """Handles plugin installation and setup."""
    
    def __init__(self):
        self.plugin_name = PLUGIN_NAME
        self.plugin_version = PLUGIN_VERSION
        self.home_dir = Path.home()
        self.flika_dir = self.home_dir / ".FLIKA"
        self.plugins_dir = self.flika_dir / "plugins"
        self.plugin_dir = self.plugins_dir / self.plugin_name
        
        # Current script directory
        self.script_dir = Path(__file__).parent.absolute()
        
        # Installation status
        self.installation_log = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log installation messages."""
        log_entry = f"[{level}] {message}"
        self.installation_log.append(log_entry)
        print(log_entry)
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        python_version = sys.version_info
        min_version = (3, 7)
        
        if python_version >= min_version:
            self.log(f"✓ Python version {python_version.major}.{python_version.minor}.{python_version.micro} is compatible")
            return True
        else:
            self.log(f"✗ Python version {python_version.major}.{python_version.minor} is too old. Minimum required: {min_version[0]}.{min_version[1]}", "ERROR")
            return False
    
    def check_flika_installation(self) -> bool:
        """Check if FLIKA is installed and accessible."""
        try:
            import flika
            flika_version = getattr(flika, '__version__', 'unknown')
            self.log(f"✓ FLIKA found (version: {flika_version})")
            
            # Check if FLIKA directory exists
            if not self.flika_dir.exists():
                self.log("Creating FLIKA user directory...")
                self.flika_dir.mkdir(parents=True, exist_ok=True)
                
            if not self.plugins_dir.exists():
                self.log("Creating FLIKA plugins directory...")
                self.plugins_dir.mkdir(parents=True, exist_ok=True)
            
            return True
            
        except ImportError:
            self.log("✗ FLIKA not found. Please install FLIKA first.", "ERROR")
            self.log("  Visit: https://github.com/flika-org/flika", "INFO")
            return False
    
    def check_dependencies(self) -> Tuple[List[str], List[str]]:
        """Check required and optional dependencies."""
        missing_required = []
        missing_optional = []
        
        self.log("Checking dependencies...")
        
        # Check required dependencies
        for package, version_req in REQUIRED_DEPENDENCIES.items():
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                self.log(f"✓ {package} {version} found")
            except ImportError:
                self.log(f"✗ Required dependency missing: {package} {version_req}", "WARNING")
                missing_required.append(package)
        
        # Check optional dependencies
        for package, version_req in OPTIONAL_DEPENDENCIES.items():
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                self.log(f"✓ {package} {version} found")
            except ImportError:
                self.log(f"○ Optional dependency missing: {package} {version_req}", "INFO")
                missing_optional.append(package)
        
        return missing_required, missing_optional
    
    def install_dependencies(self, packages: List[str], optional: bool = False) -> bool:
        """Install missing dependencies using pip."""
        if not packages:
            return True
        
        dep_type = "optional" if optional else "required"
        self.log(f"Installing {dep_type} dependencies: {', '.join(packages)}")
        
        try:
            for package in packages:
                if optional:
                    # Ask user for optional dependencies
                    response = input(f"Install optional dependency {package}? (y/n): ").lower().strip()
                    if response not in ['y', 'yes']:
                        self.log(f"Skipping {package}")
                        continue
                
                self.log(f"Installing {package}...")
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.log(f"✓ {package} installed successfully")
                else:
                    self.log(f"✗ Failed to install {package}: {result.stderr}", "ERROR")
                    if not optional:
                        return False
            
            return True
            
        except Exception as e:
            self.log(f"✗ Error during dependency installation: {str(e)}", "ERROR")
            return False
    
    def copy_plugin_files(self) -> bool:
        """Copy plugin files to FLIKA plugins directory."""
        try:
            # Create plugin directory
            if self.plugin_dir.exists():
                response = input(f"Plugin directory exists. Overwrite? (y/n): ").lower().strip()
                if response in ['y', 'yes']:
                    shutil.rmtree(self.plugin_dir)
                else:
                    self.log("Installation cancelled by user")
                    return False
            
            self.plugin_dir.mkdir(parents=True, exist_ok=True)
            self.log(f"Created plugin directory: {self.plugin_dir}")
            
            # Files to copy
            plugin_files = [
                '__init__.py',
                'utils.py',
                'advanced_plots.py',
                'info.xml',
                'about.html',
                'README.md',
                'config.json',
                'examples.py',
                'test_tracking_plotter.py'
            ]
            
            # Copy files
            copied_files = 0
            for filename in plugin_files:
                source_file = self.script_dir / filename
                
                if source_file.exists():
                    dest_file = self.plugin_dir / filename
                    shutil.copy2(source_file, dest_file)
                    self.log(f"✓ Copied {filename}")
                    copied_files += 1
                else:
                    self.log(f"○ File not found: {filename}", "WARNING")
            
            if copied_files == 0:
                self.log("✗ No plugin files found to copy", "ERROR")
                return False
            
            self.log(f"✓ Copied {copied_files} plugin files")
            return True
            
        except Exception as e:
            self.log(f"✗ Error copying plugin files: {str(e)}", "ERROR")
            return False
    
    def create_example_data(self) -> bool:
        """Create example data files for testing."""
        try:
            # Import plugin utilities to create sample data
            sys.path.insert(0, str(self.plugin_dir))
            
            from utils import generate_sample_data, create_example_csv
            
            examples_dir = self.plugin_dir / "examples"
            examples_dir.mkdir(exist_ok=True)
            
            # Create different example datasets
            examples = [
                ("basic_tracking.csv", {"n_tracks": 20, "n_frames": 100}),
                ("long_tracks.csv", {"n_tracks": 10, "n_frames": 500}),
                ("many_short_tracks.csv", {"n_tracks": 100, "n_frames": 50}),
                ("large_dataset.csv", {"n_tracks": 200, "n_frames": 200})
            ]
            
            for filename, params in examples:
                filepath = examples_dir / filename
                create_example_csv(str(filepath), **params)
                self.log(f"✓ Created example: {filename}")
            
            self.log(f"✓ Created example datasets in: {examples_dir}")
            return True
            
        except Exception as e:
            self.log(f"○ Could not create example data: {str(e)}", "WARNING")
            return True  # Non-critical failure
    
    def setup_configuration(self) -> bool:
        """Set up plugin configuration."""
        try:
            config_file = self.plugin_dir / "user_config.json"
            
            if not config_file.exists():
                # Copy default config
                default_config = self.plugin_dir / "config.json"
                if default_config.exists():
                    shutil.copy2(default_config, config_file)
                    self.log("✓ Created user configuration file")
                else:
                    self.log("○ Default config not found, creating minimal config", "WARNING")
                    minimal_config = {
                        "plugin_info": {
                            "version": self.plugin_version,
                            "installation_date": str(Path.ctime(Path.now()))
                        }
                    }
                    with open(config_file, 'w') as f:
                        json.dump(minimal_config, f, indent=2)
            
            # Create logs directory
            logs_dir = self.plugin_dir / "logs"
            logs_dir.mkdir(exist_ok=True)
            
            # Create cache directory
            cache_dir = self.plugin_dir / "cache"
            cache_dir.mkdir(exist_ok=True)
            
            self.log("✓ Plugin configuration setup complete")
            return True
            
        except Exception as e:
            self.log(f"○ Configuration setup warning: {str(e)}", "WARNING")
            return True  # Non-critical
    
    def verify_installation(self) -> bool:
        """Verify that the plugin was installed correctly."""
        try:
            # Check that plugin directory exists and has required files
            required_files = ['__init__.py', 'info.xml', 'about.html']
            
            for filename in required_files:
                filepath = self.plugin_dir / filename
                if not filepath.exists():
                    self.log(f"✗ Required file missing: {filename}", "ERROR")
                    return False
            
            # Try to import the plugin
            sys.path.insert(0, str(self.plugin_dir))
            
            try:
                import tracking_results_plotter
                self.log("✓ Plugin import successful")
            except ImportError as e:
                self.log(f"✗ Plugin import failed: {str(e)}", "ERROR")
                return False
            
            self.log("✓ Installation verification complete")
            return True
            
        except Exception as e:
            self.log(f"✗ Installation verification failed: {str(e)}", "ERROR")
            return False
    
    def install(self) -> bool:
        """Run complete installation process."""
        self.log("=" * 60)
        self.log(f"Installing {self.plugin_name} v{self.plugin_version}")
        self.log("=" * 60)
        
        # Step 1: Check system requirements
        if not self.check_python_version():
            return False
        
        if not self.check_flika_installation():
            return False
        
        # Step 2: Check dependencies
        missing_required, missing_optional = self.check_dependencies()
        
        if missing_required:
            if not self.install_dependencies(missing_required, optional=False):
                return False
        
        if missing_optional:
            self.install_dependencies(missing_optional, optional=True)
        
        # Step 3: Install plugin files
        if not self.copy_plugin_files():
            return False
        
        # Step 4: Setup configuration
        self.setup_configuration()
        
        # Step 5: Create example data
        self.create_example_data()
        
        # Step 6: Verify installation
        if not self.verify_installation():
            return False
        
        # Success!
        self.log("=" * 60)
        self.log("🎉 Installation completed successfully!")
        self.log("=" * 60)
        self.log("")
        self.log("Next steps:")
        self.log("1. Restart FLIKA")
        self.log("2. Look for 'Launch Results Plotter' in:")
        self.log("   Plugins → Tracking Analysis → Launch Results Plotter")
        self.log(f"3. Example data files are in: {self.plugin_dir / 'examples'}")
        self.log(f"4. Plugin installed to: {self.plugin_dir}")
        self.log("")
        self.log("For help and documentation:")
        self.log("- Check the about.html file in the plugin directory")
        self.log("- Run examples.py for usage demonstrations")
        self.log("- See README.md for detailed instructions")
        
        return True
    
    def uninstall(self) -> bool:
        """Uninstall the plugin."""
        self.log("Uninstalling Tracking Results Plotter...")
        
        if not self.plugin_dir.exists():
            self.log("Plugin not found - nothing to uninstall")
            return True
        
        try:
            # Ask for confirmation
            response = input(f"Remove plugin directory {self.plugin_dir}? (y/n): ").lower().strip()
            if response not in ['y', 'yes']:
                self.log("Uninstall cancelled")
                return False
            
            # Remove plugin directory
            shutil.rmtree(self.plugin_dir)
            self.log("✓ Plugin removed successfully")
            
            self.log("Plugin uninstalled. You may need to restart FLIKA.")
            return True
            
        except Exception as e:
            self.log(f"✗ Error during uninstall: {str(e)}", "ERROR")
            return False
    
    def update(self) -> bool:
        """Update the plugin to the latest version."""
        self.log("Update functionality not yet implemented")
        self.log("To update:")
        self.log("1. Download the latest plugin files")
        self.log("2. Run: python setup.py install")
        return True
    
    def save_installation_log(self):
        """Save installation log to file."""
        try:
            log_file = self.plugin_dir / "installation_log.txt"
            with open(log_file, 'w') as f:
                f.write(f"Installation log for {self.plugin_name} v{self.plugin_version}\n")
                f.write("=" * 60 + "\n\n")
                for entry in self.installation_log:
                    f.write(entry + "\n")
            
            self.log(f"Installation log saved to: {log_file}")
            
        except Exception as e:
            self.log(f"Could not save installation log: {str(e)}", "WARNING")


class PluginManager:
    """Manage multiple plugin operations."""
    
    def __init__(self):
        self.installer = PluginInstaller()
    
    def run_tests(self) -> bool:
        """Run plugin test suite."""
        print("Running plugin test suite...")
        
        try:
            # Import and run tests
            test_file = self.installer.plugin_dir / "test_tracking_plotter.py"
            
            if not test_file.exists():
                print("Test file not found. Run installation first.")
                return False
            
            # Run tests
            result = subprocess.run([
                sys.executable, str(test_file), 'all'
            ], capture_output=True, text=True)
            
            print(result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"Error running tests: {str(e)}")
            return False
    
    def run_examples(self) -> bool:
        """Run plugin examples."""
        print("Running plugin examples...")
        
        try:
            examples_file = self.installer.plugin_dir / "examples.py"
            
            if not examples_file.exists():
                print("Examples file not found. Run installation first.")
                return False
            
            # Run examples
            result = subprocess.run([
                sys.executable, str(examples_file), 'all'
            ], capture_output=True, text=True)
            
            print(result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"Error running examples: {str(e)}")
            return False
    
    def check_status(self):
        """Check plugin installation status."""
        installer = self.installer
        
        print("Tracking Results Plotter - Status Check")
        print("=" * 50)
        
        # Check installation
        if installer.plugin_dir.exists():
            print(f"✓ Plugin installed at: {installer.plugin_dir}")
            
            # Check version
            try:
                sys.path.insert(0, str(installer.plugin_dir))
                import tracking_results_plotter
                version = getattr(tracking_results_plotter, '__version__', 'unknown')
                print(f"✓ Plugin version: {version}")
            except:
                print("○ Plugin installed but import failed")
            
        else:
            print("✗ Plugin not installed")
            return
        
        # Check FLIKA
        try:
            import flika
            print(f"✓ FLIKA available (version: {getattr(flika, '__version__', 'unknown')})")
        except ImportError:
            print("✗ FLIKA not available")
        
        # Check dependencies
        print("\nDependency Status:")
        all_deps = {**REQUIRED_DEPENDENCIES, **OPTIONAL_DEPENDENCIES}
        
        for package, version_req in all_deps.items():
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                dep_type = "Required" if package in REQUIRED_DEPENDENCIES else "Optional"
                print(f"✓ {package} {version} ({dep_type})")
            except ImportError:
                dep_type = "Required" if package in REQUIRED_DEPENDENCIES else "Optional"
                print(f"✗ {package} missing ({dep_type})")


def main():
    """Main entry point for setup script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Setup script for Tracking Results Plotter FLIKA plugin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup.py install          # Install the plugin
  python setup.py uninstall        # Remove the plugin
  python setup.py status           # Check installation status
  python setup.py test             # Run test suite
  python setup.py examples         # Run example scripts
        """
    )
    
    parser.add_argument('command', 
                       choices=['install', 'uninstall', 'update', 'status', 'test', 'examples'],
                       help='Command to execute')
    
    parser.add_argument('--force', action='store_true',
                       help='Force operation without confirmation prompts')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Create plugin manager
    manager = PluginManager()
    
    try:
        if args.command == 'install':
            success = manager.installer.install()
            if success:
                manager.installer.save_installation_log()
            sys.exit(0 if success else 1)
            
        elif args.command == 'uninstall':
            success = manager.installer.uninstall()
            sys.exit(0 if success else 1)
            
        elif args.command == 'update':
            success = manager.installer.update()
            sys.exit(0 if success else 1)
            
        elif args.command == 'status':
            manager.check_status()
            sys.exit(0)
            
        elif args.command == 'test':
            success = manager.run_tests()
            sys.exit(0 if success else 1)
            
        elif args.command == 'examples':
            success = manager.run_examples()
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()