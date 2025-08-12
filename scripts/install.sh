#!/bin/bash

# DataMax Installation Script
# This script installs DataMax and its dependencies

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
check_python() {
    log_info "Checking Python version..."
    
    if ! command_exists python3; then
        log_error "Python 3 is not installed. Please install Python 3.10 or higher."
        exit 1
    fi
    
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    required_version="3.10"
    
    if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
        log_error "Python $python_version is installed, but DataMax requires Python $required_version or higher."
        exit 1
    fi
    
    log_success "Python $python_version is installed and compatible."
}

# Check pip
check_pip() {
    log_info "Checking pip..."
    
    if ! command_exists pip3; then
        log_error "pip3 is not installed. Please install pip."
        exit 1
    fi
    
    log_success "pip is available."
}

# Create virtual environment
create_venv() {
    log_info "Creating virtual environment..."
    
    if [ -d "venv" ]; then
        log_warning "Virtual environment already exists. Removing..."
        rm -rf venv
    fi
    
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    log_success "Virtual environment created and activated."
}

# Install dependencies
install_dependencies() {
    log_info "Installing dependencies..."
    
    # Install core dependencies
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        log_success "Core dependencies installed."
    else
        log_warning "requirements.txt not found. Installing from PyPI..."
        pip install datamax
    fi
    
    # Install development dependencies if requested
    if [ "$INSTALL_DEV" = "true" ] && [ -f "dev-requirements.txt" ]; then
        log_info "Installing development dependencies..."
        pip install -r dev-requirements.txt
        log_success "Development dependencies installed."
    fi
    
    # Install test dependencies if requested
    if [ "$INSTALL_TEST" = "true" ] && [ -f "test-requirements.txt" ]; then
        log_info "Installing test dependencies..."
        pip install -r test-requirements.txt
        log_success "Test dependencies installed."
    fi
}

# Install DataMax
install_datamax() {
    log_info "Installing DataMax..."
    
    if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
        # Install from source
        pip install -e .
        log_success "DataMax installed from source."
    else
        # Install from PyPI
        pip install datamax
        log_success "DataMax installed from PyPI."
    fi
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    if python -c "import datamax; print(f'DataMax {datamax.__version__} installed successfully')" 2>/dev/null; then
        log_success "DataMax installation verified."
    else
        log_error "DataMax installation verification failed."
        exit 1
    fi
}

# Setup configuration
setup_config() {
    log_info "Setting up configuration..."
    
    # Create config directory
    mkdir -p ~/.datamax
    
    # Create default config if it doesn't exist
    if [ ! -f "~/.datamax/config.yaml" ]; then
        cat > ~/.datamax/config.yaml << EOF
# DataMax Configuration
logging:
  level: INFO
  format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

crawler:
  max_concurrent: 10
  delay: 1.0
  timeout: 30.0
  retries: 3

storage:
  type: local
  path: ./data

parser:
  output_format: json
  clean_text: true
EOF
        log_success "Default configuration created at ~/.datamax/config.yaml"
    else
        log_info "Configuration already exists at ~/.datamax/config.yaml"
    fi
}

# Create directories
create_directories() {
    log_info "Creating data directories..."
    
    mkdir -p data output logs cache
    
    log_success "Data directories created."
}

# Install pre-commit hooks
install_hooks() {
    if [ "$INSTALL_DEV" = "true" ] && [ -f ".pre-commit-config.yaml" ]; then
        log_info "Installing pre-commit hooks..."
        
        if command_exists pre-commit; then
            pre-commit install
            log_success "Pre-commit hooks installed."
        else
            log_warning "pre-commit not found. Skipping hook installation."
        fi
    fi
}

# Print usage information
print_usage() {
    cat << EOF

${GREEN}DataMax Installation Complete!${NC}

To get started:

1. Activate the virtual environment:
   ${BLUE}source venv/bin/activate${NC}

2. Check the installation:
   ${BLUE}datamax --version${NC}

3. View help:
   ${BLUE}datamax --help${NC}

4. Run a simple crawl:
   ${BLUE}datamax crawl arxiv --query "machine learning" --max-results 5${NC}

5. Parse crawled data:
   ${BLUE}datamax parse --input data/ --output output/ --format json${NC}

Configuration file: ~/.datamax/config.yaml
Data directory: ./data
Output directory: ./output
Logs directory: ./logs

For more information, visit: https://github.com/Hi-Dolphin/datamax

EOF
}

# Main installation function
main() {
    log_info "Starting DataMax installation..."
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dev)
                INSTALL_DEV="true"
                shift
                ;;
            --test)
                INSTALL_TEST="true"
                shift
                ;;
            --no-venv)
                NO_VENV="true"
                shift
                ;;
            --help)
                echo "Usage: $0 [--dev] [--test] [--no-venv] [--help]"
                echo "  --dev      Install development dependencies"
                echo "  --test     Install test dependencies"
                echo "  --no-venv  Skip virtual environment creation"
                echo "  --help     Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Check prerequisites
    check_python
    check_pip
    
    # Create virtual environment unless skipped
    if [ "$NO_VENV" != "true" ]; then
        create_venv
    fi
    
    # Install dependencies and DataMax
    install_dependencies
    install_datamax
    
    # Verify installation
    verify_installation
    
    # Setup configuration and directories
    setup_config
    create_directories
    
    # Install development tools
    install_hooks
    
    # Print usage information
    print_usage
    
    log_success "DataMax installation completed successfully!"
}

# Run main function
main "$@"