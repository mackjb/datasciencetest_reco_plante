#!/usr/bin/env python3
"""
Script de test GPU pour devcontainer CUDA 12.8 + TensorFlow
"""

import sys

print("\n" + "="*60)
print("  TEST GPU - RTX 5070 CUDA 12.8")
print("="*60 + "\n")

# Test TensorFlow
try:
    import tensorflow as tf
    
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    
    # Liste des GPU
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\nGPUs d\u00e9tect\u00e9s: {len(gpus)}")
    
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            try:
                details = tf.config.experimental.get_device_details(gpu)
                print(f"    Compute Capability: {details.get('compute_capability', 'N/A')}")
            except:
                pass
        
        # Test de calcul sur GPU:0
        print("\nTest matmul sur /GPU:0...")
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
        
        print(f"\u2705 Calcul r\u00e9ussi! Shape: {c.shape}")
        print("\n\u2705 GPU fonctionnel!")
    else:
        print("\u274c Aucun GPU d\u00e9tect\u00e9!")
        print("V\u00e9rifiez:")
        print("  - Drivers NVIDIA install\u00e9s")
        print("  - Docker avec --gpus=all")
        print("  - CUDA 12.8 compatible")
        sys.exit(1)
        
except ImportError as e:
    print(f"\u274c TensorFlow non install\u00e9: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\u274c Erreur: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("Pour Jupyter: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root")
print("="*60 + "\n")
