# PackageManager - FLIKA Plugin

A FLIKA plugin for managing Python package installations and dependencies. This plugin provides a graphical interface for viewing installed packages, installing new packages, and managing package versions within your FLIKA environment.

## Features

### Package Management
- View currently installed packages
- Install default package sets
- Install individual packages
- Version control for packages
- Support for both conda and pip installations

### Installation Methods
- Batch installation of predefined package sets
- Single package installation with version constraints
- Fallback installation paths (conda → pip)
- Version range specification support

### User Interface
- Package list viewer
- Installation dialog
- Version range selector
- Package selection from predefined list
- Manual package name entry

## Installation

### Prerequisites
- FLIKA (version >= 0.1.0)
- Conda environment
- Python package management tools:
  - conda
  - pip

### Installing the Plugin
1. Clone this repository into your FLIKA plugins directory:
```bash
cd ~/.FLIKA/plugins
git clone https://github.com/yourusername/packageManager.git
```

2. Restart FLIKA to load the plugin

## Usage

### Viewing Installed Packages
1. Launch FLIKA
2. Navigate to the PackageManager plugin
3. Click 'Show Packages' to view currently installed packages

### Installing Default Package Set
1. Click 'Install Packages'
2. Confirm installation in dialog
3. Wait for installation to complete
4. Restart FLIKA to apply changes

### Installing Single Package
1. Click 'Start Dialog'
2. Choose package:
   - Select from list
   - Enter package name manually
3. Specify version constraints (optional):
   - Minimum version
   - Maximum version
4. Click 'Go' to install

## Package Version Specification

### Version Format Options
- Exact version: `package==1.2.3`
- Minimum version: `package>=1.2.3`
- Maximum version: `package<=1.2.3`
- Version range: `package>=1.2.3,<2.0.0`
- Latest version: `package`

## Implementation Details

### Installation Process
1. Attempts conda installation first
2. Falls back to pip if conda fails
3. Tries version-free installation if versioned install fails
4. Records successful and failed installations

### Environment Detection
- Automatically detects conda environment
- Uses appropriate package manager
- Maintains installation logs

## Version History

Current Version: 2020.07.26

## Author

George Dickinson

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Notes

- Always review package list before batch installation
- Some packages may require manual conflict resolution
- Restart FLIKA after installing new packages
- Installation time varies with package size and dependencies
- Internet connection required for package installation