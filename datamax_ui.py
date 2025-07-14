#!/usr/bin/env python3
"""
DataMax Gradio Web Interface
A comprehensive web UI for DataMax file parsing, cleaning, and AI annotation.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import gradio as gr

import datamax
from datamax.exceptions import DataMaxError


# Global state management
class UIState:
    def __init__(self):
        self.parsed_results = {}
        self.processing_history = []
        self.current_session = str(int(time.time()))
        
    def add_result(self, file_name: str, result: dict):
        self.parsed_results[file_name] = result
        self.processing_history.append({
            "timestamp": time.time(),
            "file_name": file_name,
            "status": "success",
            "content_length": len(result.get("content", ""))
        })
    
    def get_stats(self):
        total_files = len(self.processing_history)
        success_files = len([h for h in self.processing_history if h["status"] == "success"])
        total_content = sum(h.get("content_length", 0) for h in self.processing_history)
        
        return {
            "total_files": total_files,
            "success_rate": (success_files / total_files * 100) if total_files > 0 else 0,
            "total_content_length": total_content
        }


ui_state = UIState()


# Utility functions
def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024
        i += 1
    return f"{size_bytes:.1f} {size_names[i]}"


def create_processing_chart():
    """Create processing statistics chart."""
    if not ui_state.processing_history:
        return None
    
    df = pd.DataFrame(ui_state.processing_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    fig = px.line(df, x='timestamp', y='content_length', 
                  title='Processing Timeline',
                  labels={'content_length': 'Content Length (chars)', 'timestamp': 'Time'})
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


# File parsing functions
def parse_single_file(
    file_obj,
    use_mineru: bool = False,
    to_markdown: bool = False,
    domain: str = "Technology",
    ttl_cache: int = 3600
) -> Tuple[str, str, str]:
    """Parse a single uploaded file."""
    if file_obj is None:
        return "❌ No file uploaded", "", ""
    
    try:
        # Create client with configuration
        client = datamax.DataMaxClient(
            domain=domain,
            parse={
                "use_mineru": use_mineru,
                "to_markdown": to_markdown,
                "ttl_cache": ttl_cache
            }
        )
        
        # Parse the file
        result = client.parse_file(file_obj.name)
        
        # Store result
        file_name = os.path.basename(file_obj.name)
        ui_state.add_result(file_name, result)
        
        # Prepare output
        content = result.get("content", "")
        metadata = {
            "file_name": file_name,
            "file_size": format_file_size(os.path.getsize(file_obj.name)),
            "content_length": len(content),
            "domain": domain,
            "use_mineru": use_mineru,
            "to_markdown": to_markdown
        }
        
        metadata_str = json.dumps(metadata, indent=2, ensure_ascii=False)
        
        success_msg = f"✅ Successfully parsed: {file_name}\n📊 Content length: {len(content):,} characters"
        
        return success_msg, content, metadata_str
        
    except Exception as e:
        error_msg = f"❌ Error parsing file: {str(e)}"
        return error_msg, "", ""


def clean_text_data(
    content: str,
    methods: List[str],
    show_diff: bool = True
) -> Tuple[str, str, str]:
    """Clean text data with selected methods."""
    if not content.strip():
        return "❌ No content to clean", "", ""
    
    try:
        client = datamax.DataMaxClient()
        
        # Perform cleaning
        cleaned_result = client.clean_data(
            content=content,
            methods=methods
        )
        
        if isinstance(cleaned_result, dict):
            cleaned_content = cleaned_result.get("content", "")
        else:
            cleaned_content = cleaned_result
        
        # Calculate statistics
        original_length = len(content)
        cleaned_length = len(cleaned_content)
        reduction_percent = ((original_length - cleaned_length) / original_length * 100) if original_length > 0 else 0
        
        stats = {
            "original_length": original_length,
            "cleaned_length": cleaned_length,
            "reduction_percent": reduction_percent,
            "methods_used": methods
        }
        
        stats_str = json.dumps(stats, indent=2, ensure_ascii=False)
        
        # Create diff view if requested
        diff_view = ""
        if show_diff and len(content) < 5000:  # Only for shorter texts
            diff_view = f"📝 **Original** ({original_length:,} chars):\n{content[:500]}...\n\n"
            diff_view += f"🧹 **Cleaned** ({cleaned_length:,} chars):\n{cleaned_content[:500]}...\n\n"
            diff_view += f"📊 **Reduction**: {reduction_percent:.1f}%"
        
        success_msg = f"✅ Cleaning completed\n📊 Reduced by {reduction_percent:.1f}% ({original_length:,} → {cleaned_length:,} chars)"
        
        return success_msg, cleaned_content, stats_str if not show_diff else diff_view
        
    except Exception as e:
        return f"❌ Cleaning failed: {str(e)}", "", ""


def generate_ai_annotations(
    content: str,
    api_key: str,
    base_url: str,
    model_name: str = "gpt-3.5-turbo",
    question_number: int = 5,
    language: str = "zh",
    chunk_size: int = 500,
    progress=gr.Progress()
) -> Tuple[str, str, str]:
    """Generate AI annotations for content."""
    if not content.strip():
        return "❌ No content provided", "", ""
    
    if not api_key or not base_url:
        return "❌ API key and base URL are required", "", ""
    
    try:
        progress(0, desc="Initializing AI client...")
        
        client = datamax.DataMaxClient(
            ai={
                "api_key": api_key,
                "base_url": base_url,
                "model_name": model_name,
                "language": language
            }
        )
        
        progress(0.2, desc="Starting annotation generation...")
        
        # Generate annotations
        annotations = client.annotate(
            content=content,
            question_number=question_number,
            chunk_size=chunk_size
        )
        
        progress(0.8, desc="Processing results...")
        
        # Format results for display
        if annotations:
            # Create DataFrame for table display
            df_data = []
            for i, qa in enumerate(annotations, 1):
                df_data.append({
                    "ID": i,
                    "Question": qa.get("instruction", qa.get("question", "N/A")),
                    "Answer": qa.get("output", "N/A"),
                    "Label": qa.get("label", "N/A")
                })
            
            df = pd.DataFrame(df_data)
            
            # Create JSON output
            json_output = json.dumps(annotations, indent=2, ensure_ascii=False)
            
            progress(1.0, desc="Completed!")
            
            success_msg = f"✅ Generated {len(annotations)} QA pairs\n🤖 Model: {model_name}\n🌐 Language: {language}"
            
            return success_msg, df, json_output
        else:
            return "⚠️ No annotations generated", pd.DataFrame(), ""
            
    except Exception as e:
        return f"❌ AI annotation failed: {str(e)}", pd.DataFrame(), ""


def batch_process_files(
    files: List,
    processing_steps: List[str],
    clean_methods: List[str],
    ai_config: Dict[str, Any],
    progress=gr.Progress()
) -> Tuple[str, str]:
    """Process multiple files in batch."""
    if not files:
        return "❌ No files provided", ""
    
    results = []
    total_files = len(files)
    
    for i, file_obj in enumerate(files):
        try:
            progress((i + 1) / total_files, desc=f"Processing {file_obj.name}...")
            
            client = datamax.DataMaxClient()
            
            # Parse file
            if "parse" in processing_steps:
                result = client.parse_file(file_obj.name)
                content = result.get("content", "")
            else:
                continue
            
            # Clean data
            if "clean" in processing_steps and clean_methods:
                content = client.clean_data(content, methods=clean_methods)
                if isinstance(content, dict):
                    content = content.get("content", "")
            
            # AI annotation
            annotations = []
            if "annotate" in processing_steps and ai_config.get("api_key"):
                try:
                    annotations = client.annotate(
                        content=content,
                        api_key=ai_config["api_key"],
                        base_url=ai_config["base_url"],
                        model_name=ai_config.get("model_name", "gpt-3.5-turbo")
                    )
                except Exception as e:
                    annotations = [{"error": str(e)}]
            
            results.append({
                "file_name": os.path.basename(file_obj.name),
                "status": "success",
                "content_length": len(content),
                "annotations_count": len(annotations),
                "content": content[:200] + "..." if len(content) > 200 else content
            })
            
        except Exception as e:
            results.append({
                "file_name": os.path.basename(file_obj.name),
                "status": "failed",
                "error": str(e)
            })
    
    # Create summary
    success_count = len([r for r in results if r["status"] == "success"])
    summary = f"✅ Processed {success_count}/{total_files} files successfully"
    
    # Create results table
    df = pd.DataFrame(results)
    results_json = json.dumps(results, indent=2, ensure_ascii=False)
    
    return summary, results_json


# Create the Gradio interface
def create_datamax_interface():
    """Create the main DataMax Gradio interface."""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .tab-nav {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    .gr-button-primary {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        border: none;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    """
    
    with gr.Blocks(css=custom_css, title="DataMax - File Processing Toolkit") as demo:
        
        # Header
        gr.Markdown("""
        # 🚀 DataMax - Advanced File Processing Toolkit
        
        **Powerful multi-format file parsing, intelligent data cleaning, and AI-powered annotation**
        
        > 💡 **Quick Start**: Upload a file → Configure options → Process → Download results
        """)
        
        # Statistics display
        with gr.Row():
            with gr.Column(scale=1):
                stats_display = gr.JSON(
                    label="📊 Session Statistics",
                    value=ui_state.get_stats()
                )
            with gr.Column(scale=2):
                processing_chart = gr.Plot(
                    label="📈 Processing Timeline",
                    value=create_processing_chart()
                )
        
        # Main interface tabs
        with gr.Tabs():
            
            # Tab 1: File Parsing
            with gr.Tab("📄 File Parsing", elem_id="parsing-tab"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📁 Upload & Configure")
                        
                        file_input = gr.File(
                            label="Upload File",
                            file_types=[".pdf", ".docx", ".doc", ".pptx", ".ppt", 
                                       ".xlsx", ".xls", ".txt", ".md", ".html", 
                                       ".epub", ".jpg", ".jpeg", ".png", ".webp"],
                            type="filepath"
                        )
                        
                        with gr.Accordion("⚙️ Parsing Options", open=False):
                            domain_dropdown = gr.Dropdown(
                                choices=["Technology", "Finance", "Health", "Education", 
                                        "Legal", "Marketing", "Sales", "Entertainment", "Science"],
                                value="Technology",
                                label="Domain"
                            )
                            
                            use_mineru_checkbox = gr.Checkbox(
                                label="Use MinerU (PDF advanced parsing)",
                                value=False
                            )
                            
                            to_markdown_checkbox = gr.Checkbox(
                                label="Convert to Markdown",
                                value=False
                            )
                            
                            ttl_slider = gr.Slider(
                                minimum=0,
                                maximum=7200,
                                value=3600,
                                step=300,
                                label="Cache TTL (seconds)"
                            )
                        
                        parse_btn = gr.Button(
                            "🚀 Parse File",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 Results")
                        
                        parse_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            max_lines=3
                        )
                        
                        parsed_content = gr.Textbox(
                            label="Parsed Content",
                            lines=15,
                            max_lines=20,
                            interactive=False
                        )
                        
                        with gr.Accordion("📋 Metadata", open=False):
                            metadata_json = gr.JSON(label="File Metadata")
                        
                        download_parsed = gr.DownloadButton(
                            "💾 Download Parsed Content",
                            visible=False
                        )
            
            # Tab 2: Data Cleaning
            with gr.Tab("🧹 Data Cleaning", elem_id="cleaning-tab"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🛠️ Cleaning Configuration")
                        
                        cleaning_input = gr.Textbox(
                            label="Input Text (or use parsed content)",
                            lines=8,
                            placeholder="Paste your text here or parse a file first..."
                        )
                        
                        use_parsed_btn = gr.Button(
                            "📄 Use Last Parsed Content",
                            size="sm"
                        )
                        
                        cleaning_methods = gr.CheckboxGroup(
                            choices=[
                                ("🔧 Abnormal Cleaning", "abnormal"),
                                ("🛡️ Privacy Desensitization", "private"),
                                ("🔍 Text Filtering", "filter")
                            ],
                            value=["abnormal", "filter"],
                            label="Cleaning Methods"
                        )
                        
                        show_diff_checkbox = gr.Checkbox(
                            label="Show before/after comparison",
                            value=True
                        )
                        
                        clean_btn = gr.Button(
                            "🧹 Clean Data",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### ✨ Cleaning Results")
                        
                        cleaning_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            max_lines=3
                        )
                        
                        cleaned_output = gr.Textbox(
                            label="Cleaned Content",
                            lines=12,
                            interactive=False
                        )
                        
                        cleaning_stats = gr.Textbox(
                            label="Cleaning Statistics",
                            lines=8,
                            interactive=False
                        )
                        
                        download_cleaned = gr.DownloadButton(
                            "💾 Download Cleaned Content",
                            visible=False
                        )
            
            # Tab 3: AI Annotation
            with gr.Tab("🤖 AI Annotation", elem_id="annotation-tab"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🔧 AI Configuration")
                        
                        annotation_input = gr.Textbox(
                            label="Content to Annotate",
                            lines=6,
                            placeholder="Input content for annotation..."
                        )
                        
                        use_cleaned_btn = gr.Button(
                            "🧹 Use Last Cleaned Content",
                            size="sm"
                        )
                        
                        with gr.Accordion("🤖 AI Settings", open=True):
                            api_key_input = gr.Textbox(
                                label="API Key",
                                type="password",
                                placeholder="sk-..."
                            )
                            
                            base_url_input = gr.Textbox(
                                label="Base URL",
                                value="https://api.openai.com/v1",
                                placeholder="https://api.openai.com/v1"
                            )
                            
                            model_dropdown = gr.Dropdown(
                                choices=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "claude-3", "custom"],
                                value="gpt-3.5-turbo",
                                label="Model"
                            )
                            
                            language_radio = gr.Radio(
                                choices=[("中文", "zh"), ("English", "en")],
                                value="zh",
                                label="Language"
                            )
                            
                            question_number_slider = gr.Slider(
                                minimum=1,
                                maximum=20,
                                value=5,
                                step=1,
                                label="Questions per Chunk"
                            )
                            
                            chunk_size_slider = gr.Slider(
                                minimum=200,
                                maximum=2000,
                                value=500,
                                step=100,
                                label="Chunk Size"
                            )
                        
                        annotate_btn = gr.Button(
                            "🚀 Generate Annotations",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### 🎯 Annotation Results")
                        
                        annotation_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            max_lines=3
                        )
                        
                        qa_table = gr.Dataframe(
                            label="📝 Generated QA Pairs",
                            headers=["ID", "Question", "Answer", "Label"],
                            datatype=["number", "str", "str", "str"],
                            wrap=True,
                            height=400
                        )
                        
                        with gr.Accordion("📄 JSON Export", open=False):
                            qa_json = gr.Textbox(
                                label="JSON Output",
                                lines=10,
                                interactive=False
                            )
                        
                        download_qa = gr.DownloadButton(
                            "💾 Download QA Data",
                            visible=False
                        )
            
            # Tab 4: Batch Processing
            with gr.Tab("⚡ Batch Processing", elem_id="batch-tab"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📁 Batch Configuration")
                        
                        batch_files = gr.File(
                            label="Upload Multiple Files",
                            file_count="multiple",
                            file_types=[".pdf", ".docx", ".doc", ".txt", ".md"]
                        )
                        
                        processing_steps = gr.CheckboxGroup(
                            choices=[
                                ("📄 Parse Files", "parse"),
                                ("🧹 Clean Data", "clean"),
                                ("🤖 AI Annotate", "annotate")
                            ],
                            value=["parse"],
                            label="Processing Pipeline"
                        )
                        
                        batch_clean_methods = gr.CheckboxGroup(
                            choices=[
                                ("🔧 Abnormal", "abnormal"),
                                ("🛡️ Privacy", "private"),
                                ("🔍 Filter", "filter")
                            ],
                            value=["abnormal"],
                            label="Cleaning Methods (if enabled)"
                        )
                        
                        with gr.Accordion("🤖 AI Config (if enabled)", open=False):
                            batch_api_key = gr.Textbox(
                                label="API Key",
                                type="password"
                            )
                            batch_base_url = gr.Textbox(
                                label="Base URL",
                                value="https://api.openai.com/v1"
                            )
                            batch_model = gr.Dropdown(
                                choices=["gpt-3.5-turbo", "gpt-4"],
                                value="gpt-3.5-turbo",
                                label="Model"
                            )
                        
                        batch_process_btn = gr.Button(
                            "⚡ Start Batch Processing",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 Processing Results")
                        
                        batch_status = gr.Textbox(
                            label="Status",
                            interactive=False
                        )
                        
                        batch_results = gr.Textbox(
                            label="Results Summary",
                            lines=15,
                            interactive=False
                        )
                        
                        download_batch = gr.DownloadButton(
                            "💾 Download Batch Results",
                            visible=False
                        )
            
            # Tab 5: Statistics & History
            with gr.Tab("📊 Analytics", elem_id="analytics-tab"):
                gr.Markdown("### 📈 Processing Analytics")
                
                with gr.Row():
                    refresh_stats_btn = gr.Button("🔄 Refresh Statistics")
                    clear_history_btn = gr.Button("🗑️ Clear History", variant="stop")
                
                with gr.Row():
                    with gr.Column():
                        session_stats = gr.JSON(
                            label="📊 Current Session",
                            value=ui_state.get_stats()
                        )
                    
                    with gr.Column():
                        history_table = gr.Dataframe(
                            label="📋 Processing History",
                            headers=["Timestamp", "File", "Status", "Content Length"],
                            interactive=False
                        )
                
                analytics_chart = gr.Plot(
                    label="📈 Processing Timeline",
                    value=create_processing_chart()
                )
        
        # Event handlers
        def update_stats():
            return ui_state.get_stats(), create_processing_chart()
        
        def clear_history():
            ui_state.processing_history.clear()
            ui_state.parsed_results.clear()
            return ui_state.get_stats(), create_processing_chart(), pd.DataFrame()
        
        def use_last_parsed():
            if ui_state.parsed_results:
                latest_file = list(ui_state.parsed_results.keys())[-1]
                latest_content = ui_state.parsed_results[latest_file].get("content", "")
                return latest_content
            return "No parsed content available"
        
        def use_last_cleaned():
            # This would need to be implemented to store cleaned content
            return "Feature coming soon..."
        
        # Wire up event handlers
        parse_btn.click(
            fn=parse_single_file,
            inputs=[file_input, use_mineru_checkbox, to_markdown_checkbox, 
                   domain_dropdown, ttl_slider],
            outputs=[parse_status, parsed_content, metadata_json]
        )
        
        use_parsed_btn.click(
            fn=use_last_parsed,
            outputs=[cleaning_input]
        )
        
        clean_btn.click(
            fn=clean_text_data,
            inputs=[cleaning_input, cleaning_methods, show_diff_checkbox],
            outputs=[cleaning_status, cleaned_output, cleaning_stats]
        )
        
        annotate_btn.click(
            fn=generate_ai_annotations,
            inputs=[annotation_input, api_key_input, base_url_input, 
                   model_dropdown, question_number_slider, language_radio, chunk_size_slider],
            outputs=[annotation_status, qa_table, qa_json],
            show_progress=True
        )
        
        batch_process_btn.click(
            fn=lambda files, steps, clean_methods, api_key, base_url, model: batch_process_files(
                files, steps, clean_methods, 
                {"api_key": api_key, "base_url": base_url, "model_name": model}
            ),
            inputs=[batch_files, processing_steps, batch_clean_methods,
                   batch_api_key, batch_base_url, batch_model],
            outputs=[batch_status, batch_results],
            show_progress=True
        )
        
        refresh_stats_btn.click(
            fn=update_stats,
            outputs=[session_stats, analytics_chart]
        )
        
        clear_history_btn.click(
            fn=clear_history,
            outputs=[session_stats, analytics_chart, history_table]
        )
        
        # Auto-refresh stats periodically
        demo.load(
            fn=update_stats,
            outputs=[stats_display, processing_chart],
            every=30  # Refresh every 30 seconds
        )
    
    return demo


if __name__ == "__main__":
    # Create and launch the interface
    demo = create_datamax_interface()
    
    # Launch with configuration
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True,
        inbrowser=True,
        favicon_path=None,
        ssl_verify=False
    )