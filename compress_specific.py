import os
from PIL import Image

files_to_compress = [
    "Deep_Learning/Interpretability/gradcam_input/in_wild/1000027883.jpg",
    "Deep_Learning/Interpretability/gradcam_input/in_wild/20251013_171602.jpg",
    "results/Deep_Learning/gradcam_outputs/archi3_disease_interpretability_data/archi3_disease_cam_1000027883_gradcam_overlay.png",
    "results/Deep_Learning/gradcam_outputs/archi3_disease_interpretability_data/archi3_disease_cam_1000027883_source.png",
    "results/Deep_Learning/gradcam_outputs/archi3_disease_interpretability_data/archi3_disease_cam_20251013_171602_gradcam_overlay.png",
    "results/Deep_Learning/gradcam_outputs/archi3_disease_interpretability_data/archi3_disease_cam_20251013_171602_source.png",
    "results/Deep_Learning/gradcam_outputs/archi3_disease_interpretability_data/archi3_disease_cam_20251013_200349_gradcam_overlay.png",
    "results/Deep_Learning/gradcam_outputs/archi3_disease_interpretability_data/archi3_disease_cam_20251013_200349_source.png",
    "results/Deep_Learning/gradcam_outputs/archi3_disease_interpretability_data/archi3_disease_cam_20251013_200400_gradcam_overlay.png",
    "results/Deep_Learning/gradcam_outputs/archi3_disease_interpretability_data/archi3_disease_cam_20251013_200400_source.png",
    "results/Deep_Learning/gradcam_outputs/archi3_disease_interpretability_data/archi3_disease_cam_IMG_2884_gradcam_overlay.png",
    "results/Deep_Learning/gradcam_outputs/archi3_disease_interpretability_data/archi3_disease_cam_IMG_2884_source.png",
    "results/Deep_Learning/gradcam_outputs/archi3_disease_interpretability_data/archi3_disease_cam_IMG_2885_gradcam_overlay.png",
    "results/Deep_Learning/gradcam_outputs/archi3_disease_interpretability_data/archi3_disease_cam_IMG_2885_source.png",
    "results/Deep_Learning/gradcam_outputs/archi3_disease_interpretability_data/archi3_disease_cam_IMG_2886_gradcam_overlay.png",
    "results/Deep_Learning/gradcam_outputs/archi3_disease_interpretability_data/archi3_disease_cam_IMG_2886_source.png",
    "results/Deep_Learning/gradcam_outputs/archi3_disease_interpretability_data/archi3_disease_cam_IMG_2887_gradcam_overlay.png",
    "results/Deep_Learning/gradcam_outputs/archi3_disease_interpretability_data/archi3_disease_cam_IMG_2887_source.png",
    "results/Deep_Learning/gradcam_outputs/archi9_disease_interpretability_data/archi9_disease_cam_1000027883_gradcam_overlay.png",
    "results/Deep_Learning/gradcam_outputs/archi9_disease_interpretability_data/archi9_disease_cam_20251013_171602_gradcam_overlay.png",
    "results/Deep_Learning/gradcam_outputs/archi9_disease_interpretability_data/archi9_disease_cam_20251013_200349_gradcam_overlay.png",
    "results/Deep_Learning/gradcam_outputs/archi9_disease_interpretability_data/archi9_disease_cam_20251013_200400_gradcam_overlay.png",
    "results/Deep_Learning/gradcam_outputs/archi9_disease_interpretability_data/archi9_disease_cam_IMG_2884_gradcam_overlay.png",
    "results/Deep_Learning/gradcam_outputs/archi9_disease_interpretability_data/archi9_disease_cam_IMG_2885_gradcam_overlay.png",
    "results/Deep_Learning/gradcam_outputs/archi9_disease_interpretability_data/archi9_disease_cam_IMG_2886_gradcam_overlay.png",
    "results/Deep_Learning/gradcam_outputs/archi9_disease_interpretability_data/archi9_disease_cam_IMG_2887_gradcam_overlay.png"
]

for p in files_to_compress:
    if os.path.exists(p):
        if os.path.getsize(p) > 2 * 1024 * 1024:  # If larger than 2MB
            print(f"Compressing {p}...")
            try:
                img = Image.open(p)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Resize if too big
                if img.height > 1080 or img.width > 1920:
                     img.thumbnail((1920, 1080))
                
                img.save(p, optimize=True, quality=70)
                print(f" -> New size: {os.path.getsize(p)/1024/1024:.2f} MB")
            except Exception as e:
                print(f"Error on {p}: {e}")
    else:
        print(f"File not found: {p}")
