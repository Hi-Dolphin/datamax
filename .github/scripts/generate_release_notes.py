#!/usr/bin/env python3
"""
AI智能生成GitHub Release发布说明
用于DataMax项目的版本发布自动化
"""

import json
import sys
import os
import requests
from datetime import datetime


def summarize_with_openai(changes_text, api_key):
    """使用OpenAI API进行总结"""
    try:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        prompt = f"""请作为一个专业的软件发布经理，分析以下Git变更信息，为DataMax（一个Python数据处理工具包）生成专业的版本发布说明。

请用中文输出，格式要求：
1. 📋 **版本亮点** - 用1-2句话概括本次更新的主要特性
2. ✨ **新增功能** - 列出新功能，用emoji标记
3. 🐛 **问题修复** - 列出修复的问题
4. 🔧 **改进优化** - 列出性能和体验改进
5. 📚 **文档更新** - 文档相关变更
6. 🧪 **测试相关** - 测试相关变更

Git变更信息：
{changes_text}

请生成专业、清晰、用户友好的发布说明，重点突出对用户的价值。"""

        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "你是一个专业的软件发布经理，擅长将技术变更转换为用户友好的发布说明。"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.3
        }
        
        response = requests.post('https://api.openai.com/v1/chat/completions', 
                               headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            return None
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return None


def summarize_with_gemini(changes_text, api_key):
    """使用Google Gemini 2.5 Flash API进行总结"""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={api_key}"
        
        prompt = f"""请作为专业的软件发布经理，分析以下Git变更信息，为DataMax（一个Python数据处理工具包）生成专业的版本发布说明。

请用中文输出，格式要求：
1. 📋 **版本亮点** - 用1-2句话概括本次更新的主要特性和价值
2. ✨ **新增功能** - 列出新功能，重点说明对用户的价值
3. 🐛 **问题修复** - 列出修复的问题，说明对用户体验的改善
4. 🔧 **改进优化** - 列出性能和体验改进
5. 📚 **文档更新** - 文档相关变更（如果有）
6. 🧪 **测试相关** - 测试相关变更（如果有）

Git变更信息：
{changes_text}

请生成专业、清晰、用户友好的发布说明，重点突出对用户的价值和影响。"""

        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1500,
                "topP": 0.8,
                "topK": 40
            }
        }
        
        response = requests.post(url, json=data, timeout=45)
        
        if response.status_code == 200:
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text'].strip()
        else:
            print(f"Gemini API error: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        print(f"Gemini API error: {e}")
        return None


def create_fallback_summary(changes_text):
    """创建fallback总结"""
    lines = changes_text.split('\n')
    commits = []
    files_changed = 0
    
    for line in lines:
        if line.startswith('COMMIT:'):
            commit_msg = line.replace('COMMIT:', '').strip()
            if commit_msg and not commit_msg.startswith('Merge'):
                commits.append(commit_msg)
        elif '|' in line and ('+' in line or '-' in line):
            files_changed += 1
    
    summary = "### 📋 版本亮点\n\n"
    summary += f"本次更新包含 {len(commits)} 个提交，涉及 {files_changed} 个文件的变更。\n\n"
    
    if commits:
        summary += "### ✨ 主要变更\n\n"
        for commit in commits[:10]:  # 限制显示数量
            if any(keyword in commit.lower() for keyword in ['feat', 'add', '新增', '添加']):
                summary += f"- ✨ {commit}\n"
            elif any(keyword in commit.lower() for keyword in ['fix', '修复', 'bug']):
                summary += f"- 🐛 {commit}\n"
            elif any(keyword in commit.lower() for keyword in ['doc', '文档']):
                summary += f"- 📚 {commit}\n"
            elif any(keyword in commit.lower() for keyword in ['test', '测试']):
                summary += f"- 🧪 {commit}\n"
            elif any(keyword in commit.lower() for keyword in ['perf', '性能', '优化']):
                summary += f"- ⚡ {commit}\n"
            else:
                summary += f"- 📝 {commit}\n"
    
    return summary


def main():
    """主函数"""
    if not os.path.exists('changes_raw.txt'):
        print("No changes file found, skipping AI summary")
        return False
    
    try:
        with open('changes_raw.txt', 'r', encoding='utf-8') as f:
            changes_text = f.read()
        
        if not changes_text.strip():
            print("No changes to summarize")
            return False
        
        ai_summary = None
        
        # 优先尝试使用Gemini 2.5 Flash API（免费且性能优秀）
        gemini_key = os.getenv('GEMINI_API_KEY')
        if gemini_key:
            print("Trying Gemini 2.5 Flash API...")
            ai_summary = summarize_with_gemini(changes_text, gemini_key)
        
        # 如果Gemini失败，尝试OpenAI
        if not ai_summary:
            openai_key = os.getenv('OPENAI_API_KEY')
            if openai_key:
                print("Trying OpenAI API as fallback...")
                ai_summary = summarize_with_openai(changes_text, openai_key)
        
        # 如果AI总结失败，使用fallback
        if not ai_summary:
            print("AI APIs unavailable, using fallback summary...")
            ai_summary = create_fallback_summary(changes_text)
        
        # 保存总结结果
        with open('ai_summary.txt', 'w', encoding='utf-8') as f:
            f.write(ai_summary)
        
        print("AI summary generated successfully")
        return True
        
    except Exception as e:
        print(f"Error in AI summarization: {e}")
        # 创建基础总结
        fallback = create_fallback_summary(changes_text if 'changes_text' in locals() else "")
        with open('ai_summary.txt', 'w', encoding='utf-8') as f:
            f.write(fallback)
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 