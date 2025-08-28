#!/usr/bin/env python3
"""
Quick script to replace float() calls with _safe_float() calls in advanced_analyzer.py
"""

import re

def fix_float_calls(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace float( with _safe_float( but not inside strings or comments
    # This regex looks for float( but not when preceded by quotes or #
    pattern = r'(?<![\'"#])\bfloat\('
    fixed_content = re.sub(pattern, '_safe_float(', content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"Fixed float() calls in {file_path}")

if __name__ == "__main__":
    fix_float_calls("acla_ai_service/advanced_analyzer.py")
