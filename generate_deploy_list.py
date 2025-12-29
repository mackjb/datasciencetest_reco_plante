import re
import os

# Read the file
with open('Streamlit/tabs/deep_learning.py', 'r') as f:
    content = f.read()

# Find all paths that look like file paths
# Matches: "path/to/something.png" or "path/to/something.JPG"
# Pattern: "([^"]+\.(?:png|jpg|JPG|PNG|jpeg))"
matches = re.findall(r'"([^"]+\.(?:png|jpg|JPG|PNG|jpeg))"', content)

# Also check for single quotes
matches += re.findall(r"'([^']+\.(?:png|jpg|JPG|PNG|jpeg))'", content)

unique_paths = sorted(list(set(matches)))

print("Found asset paths:")
for p in unique_paths:
    print(p)

# Verify they exist
print("\nVerifying existence:")
missing = []
for p in unique_paths:
    if os.path.exists(p):
        print(f"[OK] {p}")
    else:
        # Some paths might be relative to Streamlit/ or execution dir. 
        # The app runs from root.
        print(f"[MISSING] {p}")
        missing.append(p)

with open('deploy_files_whitelist.txt', 'w') as f:
    for p in unique_paths:
        if p not in missing:
            f.write(p + "\n")
