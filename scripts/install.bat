@echo off
REM DataMax Installation Script for Windows
REM This script installs DataMax and its dependencies

setlocal enabledelayedexpansion

REM Colors (using ANSI escape codes if supported)
set "RED=[31m"
set "GREEN=[32m"
set "YELLOW=[33m"
set "BLUE=[34m"
set "NC=[0m"

REM Check if ANSI colors are supported
for /f "tokens=2 delims=[]" %%a in ('ver') do set winver=%%a
if "%winver:~0,2%" geq "10" (
    REM Windows 10+ supports ANSI colors
    set "COLORS_SUPPORTED=1"
) else (
    REM Older Windows versions
    set "COLORS_SUPPORTED=0"
    set "RED="
    set "GREEN="
    set "YELLOW="
    set "BLUE="
    set "NC="
)

REM Logging functions
:log_info
echo %BLUE%[INFO]%NC% %~1
goto :eof

:log_success
echo %GREEN%[SUCCESS]%NC% %~1
goto :eof

:log_warning
echo %YELLOW%[WARNING]%NC% %~1
goto :eof

:log_error
echo %RED%[ERROR]%NC% %~1
goto :eof

REM Check if command exists
:command_exists
where %1 >nul 2>&1
if %errorlevel% equ 0 (
    set "COMMAND_EXISTS=1"
) else (
    set "COMMAND_EXISTS=0"
)
goto :eof

REM Check Python version
:check_python
call :log_info "Checking Python version..."

call :command_exists python
if !COMMAND_EXISTS! equ 0 (
    call :log_error "Python is not installed or not in PATH. Please install Python 3.10 or higher."
    exit /b 1
)

for /f "tokens=*" %%a in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set python_version=%%a
set required_version=3.10

REM Simple version comparison (works for most cases)
if "%python_version%" lss "%required_version%" (
    call :log_error "Python %python_version% is installed, but DataMax requires Python %required_version% or higher."
    exit /b 1
)

call :log_success "Python %python_version% is installed and compatible."
goto :eof

REM Check pip
:check_pip
call :log_info "Checking pip..."

call :command_exists pip
if !COMMAND_EXISTS! equ 0 (
    call :log_error "pip is not installed. Please install pip."
    exit /b 1
)

call :log_success "pip is available."
goto :eof

REM Create virtual environment
:create_venv
call :log_info "Creating virtual environment..."

if exist "venv" (
    call :log_warning "Virtual environment already exists. Removing..."
    rmdir /s /q venv
)

python -m venv venv
if %errorlevel% neq 0 (
    call :log_error "Failed to create virtual environment."
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
pip install --upgrade pip setuptools wheel
if %errorlevel% neq 0 (
    call :log_error "Failed to upgrade pip."
    exit /b 1
)

call :log_success "Virtual environment created and activated."
goto :eof

REM Install dependencies
:install_dependencies
call :log_info "Installing dependencies..."

REM Install core dependencies
if exist "requirements.txt" (
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        call :log_error "Failed to install core dependencies."
        exit /b 1
    )
    call :log_success "Core dependencies installed."
) else (
    call :log_warning "requirements.txt not found. Installing from PyPI..."
    pip install datamax
    if %errorlevel% neq 0 (
        call :log_error "Failed to install DataMax from PyPI."
        exit /b 1
    )
)

REM Install development dependencies if requested
if "%INSTALL_DEV%"=="true" (
    if exist "dev-requirements.txt" (
        call :log_info "Installing development dependencies..."
        pip install -r dev-requirements.txt
        if %errorlevel% neq 0 (
            call :log_warning "Failed to install some development dependencies."
        ) else (
            call :log_success "Development dependencies installed."
        )
    )
)

REM Install test dependencies if requested
if "%INSTALL_TEST%"=="true" (
    if exist "test-requirements.txt" (
        call :log_info "Installing test dependencies..."
        pip install -r test-requirements.txt
        if %errorlevel% neq 0 (
            call :log_warning "Failed to install some test dependencies."
        ) else (
            call :log_success "Test dependencies installed."
        )
    )
)

goto :eof

REM Install DataMax
:install_datamax
call :log_info "Installing DataMax..."

if exist "setup.py" (
    REM Install from source
    pip install -e .
    if %errorlevel% neq 0 (
        call :log_error "Failed to install DataMax from source."
        exit /b 1
    )
    call :log_success "DataMax installed from source."
) else if exist "pyproject.toml" (
    REM Install from source with pyproject.toml
    pip install -e .
    if %errorlevel% neq 0 (
        call :log_error "Failed to install DataMax from source."
        exit /b 1
    )
    call :log_success "DataMax installed from source."
) else (
    REM Install from PyPI
    pip install datamax
    if %errorlevel% neq 0 (
        call :log_error "Failed to install DataMax from PyPI."
        exit /b 1
    )
    call :log_success "DataMax installed from PyPI."
)

goto :eof

REM Verify installation
:verify_installation
call :log_info "Verifying installation..."

python -c "import datamax; print(f'DataMax {datamax.__version__} installed successfully')" >nul 2>&1
if %errorlevel% neq 0 (
    call :log_error "DataMax installation verification failed."
    exit /b 1
)

call :log_success "DataMax installation verified."
goto :eof

REM Setup configuration
:setup_config
call :log_info "Setting up configuration..."

REM Create config directory
if not exist "%USERPROFILE%\.datamax" (
    mkdir "%USERPROFILE%\.datamax"
)

REM Create default config if it doesn't exist
if not exist "%USERPROFILE%\.datamax\config.yaml" (
    (
        echo # DataMax Configuration
        echo logging:
        echo   level: INFO
        echo   format: '%%(asctime^)s - %%(name^)s - %%(levelname^)s - %%(message^)s'
        echo.
        echo crawler:
        echo   max_concurrent: 10
        echo   delay: 1.0
        echo   timeout: 30.0
        echo   retries: 3
        echo.
        echo storage:
        echo   type: local
        echo   path: ./data
        echo.
        echo parser:
        echo   output_format: json
        echo   clean_text: true
    ) > "%USERPROFILE%\.datamax\config.yaml"
    call :log_success "Default configuration created at %USERPROFILE%\.datamax\config.yaml"
) else (
    call :log_info "Configuration already exists at %USERPROFILE%\.datamax\config.yaml"
)

goto :eof

REM Create directories
:create_directories
call :log_info "Creating data directories..."

if not exist "data" mkdir data
if not exist "output" mkdir output
if not exist "logs" mkdir logs
if not exist "cache" mkdir cache

call :log_success "Data directories created."
goto :eof

REM Install pre-commit hooks
:install_hooks
if "%INSTALL_DEV%"=="true" (
    if exist ".pre-commit-config.yaml" (
        call :log_info "Installing pre-commit hooks..."
        
        call :command_exists pre-commit
        if !COMMAND_EXISTS! equ 1 (
            pre-commit install
            if %errorlevel% equ 0 (
                call :log_success "Pre-commit hooks installed."
            ) else (
                call :log_warning "Failed to install pre-commit hooks."
            )
        ) else (
            call :log_warning "pre-commit not found. Skipping hook installation."
        )
    )
)
goto :eof

REM Print usage information
:print_usage
echo.
echo %GREEN%DataMax Installation Complete!%NC%
echo.
echo To get started:
echo.
echo 1. Activate the virtual environment:
echo    %BLUE%venv\Scripts\activate.bat%NC%
echo.
echo 2. Check the installation:
echo    %BLUE%datamax --version%NC%
echo.
echo 3. View help:
echo    %BLUE%datamax --help%NC%
echo.
echo 4. Run a simple crawl:
echo    %BLUE%datamax crawl arxiv --query "machine learning" --max-results 5%NC%
echo.
echo 5. Parse crawled data:
echo    %BLUE%datamax parse --input data\ --output output\ --format json%NC%
echo.
echo Configuration file: %USERPROFILE%\.datamax\config.yaml
echo Data directory: .\data
echo Output directory: .\output
echo Logs directory: .\logs
echo.
echo For more information, visit: https://github.com/Hi-Dolphin/datamax
echo.
goto :eof

REM Parse command line arguments
:parse_args
set "INSTALL_DEV=false"
set "INSTALL_TEST=false"
set "NO_VENV=false"

:parse_loop
if "%~1"=="" goto :parse_done
if "%~1"=="--dev" (
    set "INSTALL_DEV=true"
    shift
    goto :parse_loop
)
if "%~1"=="--test" (
    set "INSTALL_TEST=true"
    shift
    goto :parse_loop
)
if "%~1"=="--no-venv" (
    set "NO_VENV=true"
    shift
    goto :parse_loop
)
if "%~1"=="--help" (
    echo Usage: %0 [--dev] [--test] [--no-venv] [--help]
    echo   --dev      Install development dependencies
    echo   --test     Install test dependencies
    echo   --no-venv  Skip virtual environment creation
    echo   --help     Show this help message
    exit /b 0
)
call :log_error "Unknown option: %~1"
exit /b 1

:parse_done
goto :eof

REM Main installation function
:main
call :log_info "Starting DataMax installation..."

REM Parse command line arguments
call :parse_args %*
if %errorlevel% neq 0 exit /b %errorlevel%

REM Check prerequisites
call :check_python
if %errorlevel% neq 0 exit /b %errorlevel%

call :check_pip
if %errorlevel% neq 0 exit /b %errorlevel%

REM Create virtual environment unless skipped
if "%NO_VENV%" neq "true" (
    call :create_venv
    if %errorlevel% neq 0 exit /b %errorlevel%
)

REM Install dependencies and DataMax
call :install_dependencies
if %errorlevel% neq 0 exit /b %errorlevel%

call :install_datamax
if %errorlevel% neq 0 exit /b %errorlevel%

REM Verify installation
call :verify_installation
if %errorlevel% neq 0 exit /b %errorlevel%

REM Setup configuration and directories
call :setup_config
call :create_directories

REM Install development tools
call :install_hooks

REM Print usage information
call :print_usage

call :log_success "DataMax installation completed successfully!"
goto :eof

REM Run main function
call :main %*
exit /b %errorlevel%