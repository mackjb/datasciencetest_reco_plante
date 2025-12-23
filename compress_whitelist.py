import os
from PIL import Image

whitelist_file = 'deploy_files_whitelist.txt'

if not os.path.exists(whitelist_file):
    print("Whitelist not found!")
    exit(1)

with open(whitelist_file, 'r') as f:
    files = [l.strip('\0\n') for l in f.readlines()]

print(f"Checking {len(files)} files for compression...")

for p in files:
    if os.path.exists(p):
        if p.lower().endswith(('.png', '.jpg', '.jpeg')):
            size_mb = os.path.getsize(p) / (1024 * 1024)
            if size_mb > 1.0:  # stricter limit: 1MB
                print(f"Compressing {p} ({size_mb:.2f} MB)...")
                try:
                    img = Image.open(p)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize if huge
                    if img.height > 1080 or img.width > 1920:
                         img.thumbnail((1920, 1080))
                    
                    img.save(p, optimize=True, quality=75)
                    new_size = os.path.getsize(p) / (1024 * 1024)
                    print(f" -> Done. New size: {new_size:.2f} MB")
                except Exception as e:
                    print(f"Error compressing {p}: {e}")
            else:
                pass 
    else:
        print(f"Warning: File not found {p}")
