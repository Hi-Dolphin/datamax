#!/usr/bin/env python3
"""
DataMax UI Launcher
Checks dependencies and launches the Gradio interface.
"""

import sys
import subprocess
import importlib.util

def check_and_install_dependencies():
    """Check and install required dependencies."""
    required_packages = {
        'gradio': 'gradio>=4.0.0',
        'plotly': 'plotly>=5.17.0',
        'pandas': 'pandas>=2.0.0'
    }
    
    missing_packages = []
    
    for package, requirement in required_packages.items():
        if importlib.util.find_spec(package) is None:
            missing_packages.append(requirement)
    
    if missing_packages:
        print("🔧 Installing missing dependencies...")
        print(f"Missing: {', '.join(missing_packages)}")
        
        # Install missing packages using Tsinghua mirror
        for package in missing_packages:
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "-i", "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
                    package
                ])
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package}: {e}")
                return False
    
    return True

def main():
    """Main launcher function."""
    print("🚀 DataMax UI Launcher")
    print("=" * 50)
    
    # Check DataMax itself
    try:
        import datamax
        print(f"✅ DataMax {datamax.__version__} loaded successfully")
    except ImportError as e:
        print(f"❌ DataMax not found: {e}")
        print("Please install DataMax first: pip install -e .")
        return 1
    
    # Check and install UI dependencies
    if not check_and_install_dependencies():
        print("❌ Failed to install dependencies")
        return 1
    
    print("✅ All dependencies satisfied")
    print("🌐 Starting DataMax UI...")
    
    # Import and launch the UI
    try:
        from datamax_ui import create_datamax_interface
        
        demo = create_datamax_interface()
        
        print("🎉 DataMax UI is ready!")
        print("📱 Access the interface at: http://localhost:7860")
        print("🛑 Press Ctrl+C to stop the server")
        
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True,
            inbrowser=True,
            favicon_path=None
        )
        
    except KeyboardInterrupt:
        print("\n👋 DataMax UI stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Failed to start UI: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)