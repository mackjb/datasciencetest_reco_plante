from IPython.display import display
from src.helpers.helpers import PROJECT_ROOT
from src.data_loader.data_loader import load_plantvillage_all, load_plantvillage_five_images







if __name__ == "__main__":
    # # Test simple des fonctions de chargement
    # print("Test de load_plantvillage_all()...")
    # df_all = load_plantvillage_all()
    # print(f"Images totales chargées : {len(df_all)}")
    # print(f"Nombre de classes : {df_all['label'].nunique()}")

    print("\nTest de load_plantvillage_five_images()...")
    df5 = load_plantvillage_five_images()
    print(f"Images totales chargées : {len(df5)}")
    print(f"Nombre de classes : {df5['label'].nunique()}")
    print("Aperçu (head) du DataFrame 5 images :")
    print(df5.head())
