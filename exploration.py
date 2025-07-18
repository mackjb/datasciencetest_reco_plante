from IPython.display import display
from src.helpers.helpers import PROJECT_ROOT
from src.data_loader.data_loader import load_plantvillage_all, load_plantvillage_five_images
import time







if __name__ == "__main__":
    # print("\nTest de load_plantvillage_all()...")
    # t0 = time.time()
    # df_all = load_plantvillage_all()
    # dt0 = time.time() - t0
    # print(f"load_all: {len(df_all)} images en {dt0:.2f}s, classes: {df_all['label'].nunique()}")

    print("\nTest de load_plantvillage_five_images()...")
    t1 = time.time()
    df5 = load_plantvillage_five_images()
    dt1 = time.time() - t1
    print(f"load_5_images: {len(df5)} images en {dt1:.2f}s, classes: {df5['label'].nunique()}")
    print("Aperçu (head) du DataFrame 5 images :")
    print(df5.head())
