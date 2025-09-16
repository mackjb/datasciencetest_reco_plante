import requests
import json

def test_api():
    # URL de base de l'API (à adapter selon votre configuration)
    BASE_URL = "http://localhost:8001"
    
    # Tester l'endpoint racine
    print("\n=== Test de l'endpoint racine ===")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status code: {response.status_code}")
    print(f"Réponse: {response.json()}")
    
    # Tester l'endpoint d'information du modèle d'espèces
    print("\n=== Test du modèle d'espèces ===")
    response = requests.get(f"{BASE_URL}/model_info/especes")
    print(f"Status code: {response.status_code}")
    print("Informations du modèle:")
    print(json.dumps(response.json(), indent=2))
    
    # Tester l'endpoint d'information du modèle de maladies
    print("\n=== Test du modèle de maladies ===")
    response = requests.get(f"{BASE_URL}/model_info/maladies")
    print(f"Status code: {response.status_code}")
    print("Informations du modèle:")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_api()
