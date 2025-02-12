import os
print(f"File path: {__file__}")
print(f"Directory: {os.path.dirname(__file__)}")
print("File content begins here:")
print(f"Current directory: {os.getcwd()}")

def test_function():
    print("Test function works")