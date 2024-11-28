import cv2
from inference_sdk import InferenceHTTPClient

# Initialiser le client avec l'URL et la clé API
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="9FfnoJPocsgXs6BEgKwq"
)

# Définir le modèle à utiliser
model_id = "airplanes-n1kvk/1"

# Initialiser la webcam
cap = cv2.VideoCapture(0)  # 0 correspond à la webcam par défaut

if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la webcam.")
    exit()

print("Appuyez sur 'q' pour quitter.")

while True:
    # Lire une image de la webcam
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire le flux de la webcam.")
        break

    # Afficher l'image capturée
    cv2.imshow("Webcam - Appuyez sur 'q' pour quitter", frame)

    # Sauvegarder l'image temporairement pour l'envoi
    image_path = "temp_image.jpg"
    cv2.imwrite(image_path, frame)

    try:
        # Envoyer l'image à l'API Roboflow
        with open(image_path, "rb") as image_file:
            result = CLIENT.infer(image_file, model_id=model_id)

        # Afficher les résultats de détection
        print("Résultat de la détection :")
        print(result)
    except Exception as e:
        print(f"Erreur : {e}")

    # Quitter la boucle si 'q' est pressé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
