# 清洗

## 能力清单
- 异常清理：HTML 标签、不可见字符、空白/换行规整、简繁转换
- 质量过滤：重复率、长度范围、数字占比
- 隐私脱敏：IP/邮箱/手机号/QQ/身份证/银行卡（含 Luhn 校验）

## 使用方式

### 管道式（推荐）
```python
from datamax import DataMax

cleaned = DataMax(file_path="a.txt").clean_data([
    "abnormal",  # 异常清理
    "filter",    # 质量过滤
    "private"    # 隐私脱敏
])
print(cleaned["content"])  # 统一结构，含 lifecycle
```

### 细粒度类（进阶）
```python
from datamax.cleaner import AbnormalCleaner, TextFilter, PrivacyDesensitization

text = "含 <b>HTML</b> 与 个人信息：182****，test@example.com"

text = AbnormalCleaner(text).to_clean()["text"]
text = TextFilter(text).to_filter().get("text", text)
text = PrivacyDesensitization(text).to_private()["text"]
```

## 小贴士
- 建议先“异常清理”再做“过滤”，最后做“隐私脱敏”
- `clean_data` 会为输入/输出追加生命周期事件，便于审计与追踪

## 示例脚本

--8<-- "examples/scripts/clean_text.py"
