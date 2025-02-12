import os

def print_file_info(filename):
    print(f"\nChecking {filename}:")
    # Get file stats
    stats = os.stat(filename)
    print(f"File size (os.stat): {stats.st_size} bytes")
    
    # Try different encoding reads
    print("\nTrying different encodings:")
    for encoding in ['utf-8', 'ascii', 'latin1', 'utf-16']:
        try:
            with open(filename, 'r', encoding=encoding) as f:
                content = f.read()
                print(f"\n{encoding} read successful:")
                print(f"Content length: {len(content)}")
                print("First 100 characters:")
                print(content[:100])
        except Exception as e:
            print(f"{encoding} failed: {str(e)}")
    
    # Try binary read
    print("\nBinary read:")
    with open(filename, 'rb') as f:
        content = f.read()
        print(f"Binary size: {len(content)} bytes")
        print(f"First 20 bytes hex: {content[:20].hex()}")

print_file_info('agenda_prompts.py')