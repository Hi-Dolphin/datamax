#!/bin/bash
echo "==== [DataMax敏感信息及虚拟环境一键自检] ===="

echo
echo "1. 检查 .gitignore 是否包含虚拟环境与敏感文件"
grep -E "venv|\.venv|\.env|env|ENV|\.pem|\.crt|\.key" .gitignore || echo "⚠️  [警告] .gitignore 中未检测到虚拟环境或敏感文件屏蔽！"

echo
echo "2. 检查 .venv、venv 等虚拟环境目录是否已被Git跟踪"
git ls-files | grep -E "^(\.venv/|venv/|env/|ENV/)" && echo "❌ [风险] 检测到虚拟环境被Git跟踪！请立即移除并加到.gitignore" || echo "✅ 未检测到虚拟环境被跟踪"

echo
echo "3. 检查是否提交了 .env 文件"
git ls-files | grep "\.env" && echo "❌ [风险] 检测到 .env 文件被Git跟踪！" || echo "✅ 未检测到 .env 文件被跟踪"

echo
echo "4. 检查是否提交了私钥/证书/常见密钥文件"
git ls-files | grep -Ei "\.(pem|crt|key|pfx|cer)$" && echo "❌ [风险] 检测到私钥/证书类文件被Git跟踪！" || echo "✅ 未检测到私钥/证书文件被跟踪"

echo
echo "5. 代码中常见敏感关键词全局扫描（仅供参考，人工复查）"
grep -rniE 'password|passwd|secret|api[_-]?key|access[_-]?key|token|private[_-]?key' . --exclude-dir={.git,.venv,venv,env,__pycache__} --exclude="*.md" --exclude="*.txt" | head -n 30

echo
echo "6. 检查提交记录（最近50次）是否有敏感词"
git log -p -n 50 | grep -Ei 'password|passwd|secret|api[_-]?key|access[_-]?key|token|private[_-]?key' | head -n 20

echo
echo "==== 检查结束（如出现 ❌/⚠️ 请立即处理！） ===="
