#!/usr/bin/env python3
"""
Simple DataMax UI Test
Basic functionality test for the Gradio interface.
"""

def test_basic_functionality():
    """Test basic DataMax functionality without UI."""
    print("🧪 Testing DataMax Basic Functionality")
    print("=" * 50)
    
    try:
        # Test DataMax import
        import datamax
        print(f"✅ DataMax imported successfully (v{datamax.__version__})")
        
        # Test client creation
        client = datamax.DataMaxClient()
        print("✅ DataMaxClient created successfully")
        
        # Test basic configuration
        settings = datamax.get_settings()
        print("✅ Settings loaded successfully")
        
        # Test exception handling
        from datamax.exceptions import DataMaxError
        print("✅ Exception classes imported successfully")
        
        print("\n🎉 All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ui_dependencies():
    """Test UI dependencies."""
    print("\n🧪 Testing UI Dependencies")
    print("=" * 50)
    
    dependencies = {
        'gradio': None,
        'plotly': None,
        'pandas': None
    }
    
    for dep in dependencies:
        try:
            module = __import__(dep)
            version = getattr(module, '__version__', 'unknown')
            dependencies[dep] = version
            print(f"✅ {dep}: {version}")
        except ImportError:
            print(f"❌ {dep}: not available")
            dependencies[dep] = None
    
    return all(dep is not None for dep in dependencies.values())

def create_minimal_ui():
    """Create a minimal UI for testing."""
    print("\n🎨 Creating Minimal UI")
    print("=" * 50)
    
    try:
        import gradio as gr
        
        def parse_text(text):
            """Simple text processing function."""
            if not text:
                return "Please enter some text"
            
            try:
                import datamax
                client = datamax.DataMaxClient()
                
                # Simple text processing (simulate file parsing)
                result = {
                    "content": text,
                    "length": len(text),
                    "words": len(text.split()),
                    "status": "✅ Text processed successfully"
                }
                
                return f"""
Status: {result['status']}
Length: {result['length']} characters
Words: {result['words']} words

Content Preview:
{text[:200]}{'...' if len(text) > 200 else ''}
"""
            except Exception as e:
                return f"❌ Error: {str(e)}"
        
        # Create simple interface
        with gr.Blocks(title="DataMax Simple UI Test") as demo:
            gr.Markdown("# 🧪 DataMax UI Test")
            gr.Markdown("Simple interface to test DataMax functionality")
            
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        label="Input Text",
                        lines=5,
                        placeholder="Enter text to process..."
                    )
                    process_btn = gr.Button("🚀 Process Text", variant="primary")
                
                with gr.Column():
                    output = gr.Textbox(
                        label="Results",
                        lines=10,
                        interactive=False
                    )
            
            # Connect button to function
            process_btn.click(
                fn=parse_text,
                inputs=[text_input],
                outputs=[output]
            )
        
        print("✅ Minimal UI created successfully")
        return demo
        
    except Exception as e:
        print(f"❌ Failed to create UI: {e}")
        return None

def main():
    """Main test function."""
    print("🧪 DataMax UI Test Suite")
    print("=" * 60)
    
    # Test basic functionality
    if not test_basic_functionality():
        print("❌ Basic functionality test failed")
        return 1
    
    # Test UI dependencies
    if not test_ui_dependencies():
        print("❌ UI dependencies test failed") 
        print("💡 Please install missing dependencies:")
        print("   pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple gradio plotly")
        return 1
    
    # Create and test minimal UI
    demo = create_minimal_ui()
    if demo is None:
        print("❌ UI creation failed")
        return 1
    
    print("\n🎉 All tests passed!")
    print("🌐 Starting test server...")
    print("📱 Access at: http://localhost:7860")
    print("🛑 Press Ctrl+C to stop")
    
    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True,
            inbrowser=False,  # Don't auto-open browser in test
            quiet=False
        )
    except KeyboardInterrupt:
        print("\n👋 Test stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Server failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)