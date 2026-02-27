import syntax_checker

code = "print('hello')"

extensions_to_try = [".py", "python", "py", "python.py"]

for ext in extensions_to_try:
    try:
        print(f"Trying with {ext}:")
        print(syntax_checker.check_syntax(code, ext))
    except Exception as e:
        print(f"Error: {e}")
