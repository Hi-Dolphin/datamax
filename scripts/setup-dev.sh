#!/bin/bash

# DataMax Development Environment Setup Script
# This script sets up a complete development environment for DataMax

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.11"
NODE_VERSION="18"
REQUIRED_PYTHON_VERSION="3.10"
PROJECT_NAME="datamax"
VENV_NAME="venv"

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

log_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

log_debug() {
    if [ "$DEBUG" = "true" ]; then
        echo -e "${CYAN}[DEBUG]${NC} $1"
    fi
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if running on supported OS
check_os() {
    log_step "Checking operating system..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        if command_exists apt-get; then
            DISTRO="ubuntu"
        elif command_exists yum; then
            DISTRO="centos"
        elif command_exists pacman; then
            DISTRO="arch"
        else
            DISTRO="unknown"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        DISTRO="macos"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        OS="windows"
        DISTRO="windows"
    else
        OS="unknown"
        DISTRO="unknown"
    fi
    
    log_info "Detected OS: $OS ($DISTRO)"
    
    if [ "$OS" = "unknown" ]; then
        log_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
}

# Install system dependencies
install_system_deps() {
    log_step "Installing system dependencies..."
    
    case "$DISTRO" in
        ubuntu)
            log_info "Installing dependencies for Ubuntu/Debian..."
            sudo apt-get update
            sudo apt-get install -y \
                python3 python3-pip python3-venv python3-dev \
                build-essential libssl-dev libffi-dev \
                git curl wget unzip \
                sqlite3 libsqlite3-dev \
                postgresql-client libpq-dev \
                redis-tools \
                nodejs npm \
                docker.io docker-compose \
                jq tree htop \
                vim nano
            ;;
        centos)
            log_info "Installing dependencies for CentOS/RHEL..."
            sudo yum update -y
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y \
                python3 python3-pip python3-devel \
                openssl-devel libffi-devel \
                git curl wget unzip \
                sqlite sqlite-devel \
                postgresql postgresql-devel \
                redis \
                nodejs npm \
                docker docker-compose \
                jq tree htop \
                vim nano
            ;;
        arch)
            log_info "Installing dependencies for Arch Linux..."
            sudo pacman -Syu --noconfirm
            sudo pacman -S --noconfirm \
                python python-pip \
                base-devel openssl libffi \
                git curl wget unzip \
                sqlite \
                postgresql postgresql-libs \
                redis \
                nodejs npm \
                docker docker-compose \
                jq tree htop \
                vim nano
            ;;
        macos)
            log_info "Installing dependencies for macOS..."
            if ! command_exists brew; then
                log_info "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            
            brew update
            brew install \
                python@3.11 \
                git curl wget \
                sqlite \
                postgresql \
                redis \
                node \
                docker docker-compose \
                jq tree htop \
                vim nano
            ;;
        windows)
            log_warning "Windows detected. Please install dependencies manually:"
            log_info "1. Python 3.10+ from https://python.org"
            log_info "2. Git from https://git-scm.com"
            log_info "3. Node.js from https://nodejs.org"
            log_info "4. Docker Desktop from https://docker.com"
            log_info "5. PostgreSQL from https://postgresql.org"
            log_info "6. Redis from https://redis.io"
            return 0
            ;;
        *)
            log_error "Unsupported distribution: $DISTRO"
            exit 1
            ;;
    esac
    
    log_success "System dependencies installed"
}

# Check Python version
check_python() {
    log_step "Checking Python version..."
    
    if ! command_exists python3; then
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    local python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") 
    log_info "Found Python $python_version"
    
    # Simple version comparison
    if [ "$(echo "$python_version >= $REQUIRED_PYTHON_VERSION" | bc -l 2>/dev/null || echo "0")" != "1" ]; then
        # Fallback comparison for systems without bc
        local major=$(echo "$python_version" | cut -d. -f1)
        local minor=$(echo "$python_version" | cut -d. -f2)
        local req_major=$(echo "$REQUIRED_PYTHON_VERSION" | cut -d. -f1)
        local req_minor=$(echo "$REQUIRED_PYTHON_VERSION" | cut -d. -f2)
        
        if [ "$major" -lt "$req_major" ] || ([ "$major" -eq "$req_major" ] && [ "$minor" -lt "$req_minor" ]); then
            log_error "Python $REQUIRED_PYTHON_VERSION+ required, found $python_version"
            exit 1
        fi
    fi
    
    log_success "Python version is compatible"
}

# Setup virtual environment
setup_venv() {
    log_step "Setting up virtual environment..."
    
    if [ -d "$VENV_NAME" ]; then
        log_warning "Virtual environment already exists. Removing..."
        rm -rf "$VENV_NAME"
    fi
    
    python3 -m venv "$VENV_NAME"
    source "$VENV_NAME/bin/activate"
    
    # Upgrade pip and install build tools
    pip install --upgrade pip setuptools wheel
    
    log_success "Virtual environment created and activated"
}

# Install Python dependencies
install_python_deps() {
    log_step "Installing Python dependencies..."
    
    # Ensure virtual environment is activated
    if [ -z "$VIRTUAL_ENV" ]; then
        source "$VENV_NAME/bin/activate"
    fi
    
    # Install core dependencies
    if [ -f "requirements.txt" ]; then
        log_info "Installing core dependencies from requirements.txt..."
        pip install -r requirements.txt
    else
        log_warning "requirements.txt not found, installing basic dependencies..."
        pip install \
            requests beautifulsoup4 lxml \
            pandas numpy \
            pyyaml python-dotenv \
            click rich \
            sqlalchemy psycopg2-binary \
            redis celery \
            elasticsearch \
            pytest pytest-cov \
            black isort flake8 mypy \
            pre-commit
    fi
    
    # Install development dependencies
    if [ -f "dev-requirements.txt" ]; then
        log_info "Installing development dependencies..."
        pip install -r dev-requirements.txt
    else
        log_info "Installing common development tools..."
        pip install \
            pytest pytest-cov pytest-mock pytest-asyncio \
            black isort flake8 mypy bandit safety \
            pre-commit \
            sphinx sphinx-rtd-theme \
            jupyter notebook ipython \
            python-language-server[all] \
            debugpy
    fi
    
    # Install test dependencies
    if [ -f "test-requirements.txt" ]; then
        log_info "Installing test dependencies..."
        pip install -r test-requirements.txt
    fi
    
    # Install project in development mode
    if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
        log_info "Installing project in development mode..."
        pip install -e .
    fi
    
    log_success "Python dependencies installed"
}

# Setup pre-commit hooks
setup_precommit() {
    log_step "Setting up pre-commit hooks..."
    
    if [ -f ".pre-commit-config.yaml" ]; then
        # Ensure virtual environment is activated
        if [ -z "$VIRTUAL_ENV" ]; then
            source "$VENV_NAME/bin/activate"
        fi
        
        pre-commit install
        pre-commit install --hook-type commit-msg
        
        # Run pre-commit on all files to ensure everything works
        log_info "Running pre-commit on all files..."
        pre-commit run --all-files || log_warning "Some pre-commit checks failed"
        
        log_success "Pre-commit hooks installed"
    else
        log_warning ".pre-commit-config.yaml not found, skipping pre-commit setup"
    fi
}

# Setup configuration files
setup_config() {
    log_step "Setting up configuration files..."
    
    # Create config directory
    mkdir -p config
    
    # Create development configuration
    if [ ! -f "config/development.yaml" ]; then
        cat > config/development.yaml << EOF
# DataMax Development Configuration
logging:
  level: DEBUG
  format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  handlers:
    - type: console
    - type: file
      filename: logs/datamax-dev.log
      max_bytes: 10485760
      backup_count: 5

server:
  host: 127.0.0.1
  port: 8000
  workers: 1
  debug: true
  reload: true

crawler:
  max_concurrent: 5
  delay: 0.5
  timeout: 30.0
  retries: 3
  user_agent: 'DataMax-Dev/1.0'

storage:
  type: local
  path: ./data
  
parser:
  output_format: json
  clean_text: true
  extract_metadata: true

cache:
  type: file
  path: ./cache
  ttl: 1800

database:
  url: sqlite:///./datamax-dev.db
  echo: true

redis:
  url: redis://localhost:6379/0
  
elasticsearch:
  url: http://localhost:9200
  
testing:
  database_url: sqlite:///./datamax-test.db
  redis_url: redis://localhost:6379/1
EOF
        log_success "Created development configuration"
    fi
    
    # Create environment file
    if [ ! -f ".env.development" ]; then
        cat > .env.development << EOF
# DataMax Development Environment
DATAMAX_ENV=development
DATAMAX_CONFIG=config/development.yaml
DATAMAX_LOG_LEVEL=DEBUG
DATAMAX_DEBUG=true

# Database
DATABASE_URL=sqlite:///./datamax-dev.db

# Redis
REDIS_URL=redis://localhost:6379/0

# Elasticsearch
ELASTICSEARCH_URL=http://localhost:9200

# API Keys (add your keys here)
# OPENAI_API_KEY=your_openai_key
# ANTHROPIC_API_KEY=your_anthropic_key
EOF
        log_success "Created development environment file"
    fi
    
    # Create test environment file
    if [ ! -f ".env.test" ]; then
        cat > .env.test << EOF
# DataMax Test Environment
DATAMAX_ENV=test
DATAMAX_CONFIG=config/test.yaml
DATAMAX_LOG_LEVEL=WARNING
DATAMAX_DEBUG=false

# Test Database
DATABASE_URL=sqlite:///./datamax-test.db

# Test Redis
REDIS_URL=redis://localhost:6379/1
EOF
        log_success "Created test environment file"
    fi
}

# Create project directories
create_directories() {
    log_step "Creating project directories..."
    
    local dirs=(
        "data"
        "data/raw"
        "data/processed"
        "data/exports"
        "logs"
        "cache"
        "output"
        "tests/fixtures"
        "docs/source"
        "scripts"
        "config"
        ".vscode"
    )
    
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_debug "Created directory: $dir"
        fi
    done
    
    log_success "Project directories created"
}

# Setup VS Code configuration
setup_vscode() {
    log_step "Setting up VS Code configuration..."
    
    # Create VS Code settings
    if [ ! -f ".vscode/settings.json" ]; then
        cat > .vscode/settings.json << EOF
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.linting.banditEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/.pytest_cache": true,
        "**/.mypy_cache": true,
        "**/venv": true,
        "**/node_modules": true
    },
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests"
    ],
    "python.testing.unittestEnabled": false,
    "python.testing.nosetestsEnabled": false
}
EOF
        log_success "Created VS Code settings"
    fi
    
    # Create VS Code launch configuration
    if [ ! -f ".vscode/launch.json" ]; then
        cat > .vscode/launch.json << EOF
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "DataMax Server",
            "type": "python",
            "request": "launch",
            "module": "datamax",
            "args": ["serve", "--config", "config/development.yaml"],
            "console": "integratedTerminal",
            "envFile": "\${workspaceFolder}/.env.development"
        },
        {
            "name": "DataMax Crawl",
            "type": "python",
            "request": "launch",
            "module": "datamax",
            "args": ["crawl", "--help"],
            "console": "integratedTerminal",
            "envFile": "\${workspaceFolder}/.env.development"
        },
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "\${file}",
            "console": "integratedTerminal",
            "envFile": "\${workspaceFolder}/.env.development"
        },
        {
            "name": "Python: Pytest",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["-v"],
            "console": "integratedTerminal",
            "envFile": "\${workspaceFolder}/.env.test"
        }
    ]
}
EOF
        log_success "Created VS Code launch configuration"
    fi
    
    # Create VS Code extensions recommendations
    if [ ! -f ".vscode/extensions.json" ]; then
        cat > .vscode/extensions.json << EOF
{
    "recommendations": [
        "ms-python.python",
        "ms-python.flake8",
        "ms-python.mypy-type-checker",
        "ms-python.black-formatter",
        "ms-python.isort",
        "charliermarsh.ruff",
        "ms-vscode.vscode-json",
        "redhat.vscode-yaml",
        "ms-vscode.vscode-markdown",
        "eamodio.gitlens",
        "github.vscode-pull-request-github",
        "ms-vscode.test-adapter-converter",
        "littlefoxteam.vscode-python-test-adapter",
        "ms-vscode-remote.remote-containers",
        "ms-azuretools.vscode-docker"
    ]
}
EOF
        log_success "Created VS Code extensions recommendations"
    fi
}

# Setup development scripts
setup_dev_scripts() {
    log_step "Setting up development scripts..."
    
    # Create development runner script
    if [ ! -f "scripts/dev.sh" ]; then
        cat > scripts/dev.sh << 'EOF'
#!/bin/bash
# Development helper script

set -e

VENV_PATH="./venv"
CONFIG_FILE="config/development.yaml"
ENV_FILE=".env.development"

# Activate virtual environment
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Load environment variables
if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

case "$1" in
    serve|server)
        echo "Starting DataMax development server..."
        datamax serve --config "$CONFIG_FILE" --reload
        ;;
    test)
        echo "Running tests..."
        pytest tests/ -v --cov=datamax --cov-report=html
        ;;
    lint)
        echo "Running linters..."
        flake8 datamax tests
        black --check datamax tests
        isort --check-only datamax tests
        mypy datamax
        ;;
    format)
        echo "Formatting code..."
        black datamax tests
        isort datamax tests
        ;;
    clean)
        echo "Cleaning up..."
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find . -type f -name "*.pyc" -delete 2>/dev/null || true
        rm -rf .pytest_cache .mypy_cache .coverage htmlcov
        ;;
    install)
        echo "Installing dependencies..."
        pip install -r requirements.txt
        pip install -r dev-requirements.txt
        pip install -e .
        ;;
    *)
        echo "Usage: $0 {serve|test|lint|format|clean|install}"
        echo "  serve   - Start development server"
        echo "  test    - Run tests with coverage"
        echo "  lint    - Run code linters"
        echo "  format  - Format code with black and isort"
        echo "  clean   - Clean up cache files"
        echo "  install - Install/update dependencies"
        exit 1
        ;;
esac
EOF
        chmod +x scripts/dev.sh
        log_success "Created development script"
    fi
}

# Verify installation
verify_installation() {
    log_step "Verifying installation..."
    
    # Activate virtual environment
    if [ -z "$VIRTUAL_ENV" ]; then
        source "$VENV_NAME/bin/activate"
    fi
    
    # Check if DataMax can be imported
    if python -c "import datamax; print(f'DataMax {datamax.__version__} installed successfully')" 2>/dev/null; then
        log_success "DataMax installation verified"
    else
        log_warning "DataMax import failed, but dependencies are installed"
    fi
    
    # Check development tools
    local tools=("pytest" "black" "isort" "flake8" "mypy" "pre-commit")
    for tool in "${tools[@]}"; do
        if command -v "$tool" >/dev/null 2>&1; then
            log_debug "✓ $tool is available"
        else
            log_warning "✗ $tool is not available"
        fi
    done
    
    log_success "Installation verification completed"
}

# Print usage information
print_usage() {
    cat << EOF

${GREEN}🎉 DataMax Development Environment Setup Complete!${NC}

To get started:

${BLUE}1. Activate the virtual environment:${NC}
   source venv/bin/activate

${BLUE}2. Start the development server:${NC}
   ./scripts/dev.sh serve
   # or
   datamax serve --config config/development.yaml

${BLUE}3. Run tests:${NC}
   ./scripts/dev.sh test
   # or
   pytest tests/ -v

${BLUE}4. Format code:${NC}
   ./scripts/dev.sh format

${BLUE}5. Run linters:${NC}
   ./scripts/dev.sh lint

${BLUE}Development Commands:${NC}
   ./scripts/dev.sh serve    # Start development server
   ./scripts/dev.sh test     # Run tests with coverage
   ./scripts/dev.sh lint     # Run code linters
   ./scripts/dev.sh format   # Format code
   ./scripts/dev.sh clean    # Clean cache files
   ./scripts/dev.sh install  # Install/update dependencies

${BLUE}Configuration Files:${NC}
   config/development.yaml   # Development configuration
   .env.development         # Development environment variables
   .env.test               # Test environment variables

${BLUE}Project Structure:${NC}
   datamax/                # Main package
   tests/                  # Test files
   docs/                   # Documentation
   scripts/                # Development scripts
   data/                   # Data directory
   logs/                   # Log files
   cache/                  # Cache directory

${BLUE}VS Code:${NC}
   - Open the project in VS Code
   - Install recommended extensions
   - Use F5 to start debugging

${BLUE}Documentation:${NC}
   - README.md for project overview
   - CONTRIBUTING.md for contribution guidelines
   - docs/ for detailed documentation

Happy coding! 🚀

EOF
}

# Parse command line arguments
parse_args() {
    SKIP_SYSTEM_DEPS=false
    SKIP_VENV=false
    SKIP_PRECOMMIT=false
    DEBUG=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-system-deps)
                SKIP_SYSTEM_DEPS=true
                shift
                ;;
            --skip-venv)
                SKIP_VENV=true
                shift
                ;;
            --skip-precommit)
                SKIP_PRECOMMIT=true
                shift
                ;;
            --debug)
                DEBUG=true
                shift
                ;;
            --help|-h)
                cat << EOF
DataMax Development Environment Setup

Usage: $0 [OPTIONS]

Options:
  --skip-system-deps    Skip system dependency installation
  --skip-venv          Skip virtual environment creation
  --skip-precommit     Skip pre-commit hooks setup
  --debug              Enable debug output
  --help, -h           Show this help message

Examples:
  $0                           # Full setup
  $0 --skip-system-deps        # Skip system dependencies
  $0 --debug                   # Enable debug output

EOF
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
}

# Main setup function
main() {
    echo -e "${GREEN}"
    cat << "EOF"
 ____        _        __  __            
|  _ \  __ _| |_ __ _|  \/  | __ ___  __
| | | |/ _` | __/ _` | |\/| |/ _` \ \/ /
| |_| | (_| | || (_| | |  | | (_| |>  < 
|____/ \__,_|\__\__,_|_|  |_|\__,_/_/\_\
                                       
Development Environment Setup
EOF
    echo -e "${NC}"
    
    log_info "Starting DataMax development environment setup..."
    
    # Parse command line arguments
    parse_args "$@"
    
    # Check operating system
    check_os
    
    # Install system dependencies
    if [ "$SKIP_SYSTEM_DEPS" != "true" ]; then
        install_system_deps
    else
        log_info "Skipping system dependency installation"
    fi
    
    # Check Python
    check_python
    
    # Create project directories
    create_directories
    
    # Setup virtual environment
    if [ "$SKIP_VENV" != "true" ]; then
        setup_venv
        install_python_deps
    else
        log_info "Skipping virtual environment setup"
    fi
    
    # Setup configuration
    setup_config
    
    # Setup VS Code
    setup_vscode
    
    # Setup development scripts
    setup_dev_scripts
    
    # Setup pre-commit hooks
    if [ "$SKIP_PRECOMMIT" != "true" ]; then
        setup_precommit
    else
        log_info "Skipping pre-commit hooks setup"
    fi
    
    # Verify installation
    verify_installation
    
    # Print usage information
    print_usage
    
    log_success "Development environment setup completed successfully!"
}

# Handle signals
trap 'log_error "Setup interrupted"; exit 1' INT TERM

# Run main function
main "$@"