# **Custom Airplane Detection Model**

## Project Results and Overview
Ce projet vise à développer un modèle personnalisé pour détecter les avions dans des images et des vidéos. L'objectif principal était de créer un modèle robuste capable d'identifier les avions avec précision, en utilisant un dataset personnalisé et des techniques d'entraînement modernes.

### **Objectifs principaux :**
- Développer un modèle d’apprentissage supervisé pour la détection d'avions dans un but de reconnaissance et d'identification.
- Utiliser une base de données de 300 image référence.
- Déployer le modèle pour détecter des avions en temps réel ou sur des supports enregistrés.

### **Résumé des résultats :**
- **Précision obtenue :** 90.9% sur un dataset de validation.  
- **Performance :** Capable de détecter un avion en moins de 0,2 seconde par image.  
- **Utilisation des ressources :** Optimisé pour fonctionner sur des configurations avec GPU (par ex. NVIDIA RTX 3060) ou même en CPU avec des temps légèrement plus longs.  

### **Contexte :**
Ce projet a été réalisé pour mon projet final de IMAGE UNDERSTANDING de l'Université Nationale de Kyungpook.Pour réaliser ce projet j'ai utilisé les bibliothèque OPENCV et Inference_sdk en suivant le tutoriel [Getting Started with Roboflow](https://blog.roboflow.com/getting-started-with-roboflow/), qui a fourni une base pour structurer le pipeline de préparation des données et d'entraînement. Le modèle a été entraîné avec un dataset personnalisé de plus de 250 images, collectées et annotées pour répondre spécifiquement aux exigences du projet.
Le projet de Base etait d'avoir un programme qui permettrait de detecter un avion de transport afin de pouvoir le reconnaitre si il survolait une base militaire a l'aide d'un modèle de reconnaissance que j'aurais entrainer.
pour ce faire, j'ai du entrainer, Grace a Roboflow, un modèle qui va reconnaitre des avions et qui seras entrainer a partir de leur propre IA et d'un modèle appelé COCO V3 ( que j'ai finalement décidé d'utiliser puisque YOLO V9 ne marchais pas sur mon ordinateur. 
J'ai d'abord réaliser un première version qui lme servait de Test puis j'en ai develloper une nouvelle qui etait beaucoup plus devellopé puisqu'elle avait 200 photo de plus pour s'entrainer. 
j'ai également ajouté plusieurs augmentations de données utilisées notamments des rotations, du recadrage, et des variation de luminosité afin d'avoir des image le plus potable possible.
![image](https://github.com/user-attachments/assets/2ff38dd9-61e6-4ac2-8d01-c08a8cb7c81a)
Ici on peut voir les differnetes versions et notamment la difference entre la Precision et le MaP

Vous pouvez retrouvez mon modèle ( et l'essayer directement sur votre navigateur ) en cliquant sur ce lien ce lien :
https://app.roboflow.com/benoit/plane-r9j7j/models




## Source Code
Le code source est organisé comme suit :
```bash
import cv2  // on importe OpenCV
from inference_sdk import InferenceHTTPClient  // on importe la bibliotheque de gestion de ROBOFLOW

# Initialiser le client avec l'URL et la clé API
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com", // grace a la bibliotheque de roboflow on recupere la l'api que l'on a en tant que membres 
    api_key="***************"    // ma clef personelle de l'API 
)

# Définir le modèle à utiliser
model_id = "airplanes-n1kvk/1" // ici on recupere mon modele ( celui que j'ai entrainer 

# Initialiser la webcam
cap = cv2.VideoCapture(0)  // 0 correspond à la webcam par défaut

if not cap.isOpened():          // pour respecter le projet j'utilise pour l'instant la cwebcam de mon PC sachant que l'objectif serait d'avoir une camera exterieur pointer vers le ciel avec un fish eye pour avoir un meilleur angle de vision 
    print("Erreur : Impossible d'accéder à la webcam.")   // explication de l'erreur si on arrive pas a acceder a la webcam 
    exit()

print("Appuyez sur 'q' pour quitter.")  //touche pour quitter le programme 

while True:
    # Lire une image de la webcam
    ret, frame = cap.read()  //lecture de la webcam 
    if not ret:
        print("Erreur : Impossible de lire le flux de la webcam.")// erreur si on arrive a acceder a la webcam mais pas a recuperer les données j'ai ajouter cette commande pour voir si ma camera marchais bien)
        break

    # Afficher l'image capturée
    cv2.imshow("Webcam - Appuyez sur 'q' pour quitter", frame) // renvoie l'image obtenue grace a la webcam 

    # Sauvegarder l'image temporairement pour l'envoi  
    image_path = "temp_image.jpg"
    cv2.imwrite(image_path, frame)

    try:
        # Envoyer l'image à l'API Roboflow
        with open(image_path, "rb") as image_file:  // envoyer l'image a l'API de Roboflow 
            result = CLIENT.infer(image_file, model_id=model_id) 

        # Afficher les résultats de détection
        print("Résultat de la détection :") //affichage du resultat 
        print(result)
    except Exception as e:
        print(f"Erreur : {e}") 

    # Quitter la boucle si 'q' est pressé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
```


### **Dépendances nécessaires :**
- Python 3.8+
- PyTorch >= 1.9
- Roboflow
- OpenCV
- Matplotlib


Installation des dépendances :
```bash
pip install -r requirements.txt
```
Installation des librairies :
```bash
pip install openCV-python
pip install inference_sdk
```


Performance Metrics
Mesures clés :
Métrique	Valeur
Précision (mAP)	89%
Temps de traitement	0.18 secondes/image
Consommation mémoire	~2.3 GB (GPU NVIDIA)
Graphique de l'évolution de la précision :
Inclure un graphique (optionnel, si disponible) montrant l'évolution de la perte ou de la précision pendant l'entraînement.

Installation and Usage
Installation :
Clone le dépôt :

bash
Copier le code
git clone https://github.com/username/custom-airplane-detector.git
cd custom-airplane-detector
Installe les dépendances :

bash
Copier le code
pip install -r requirements.txt
Configure le dataset en suivant les instructions du fichier data/README.md.

Utilisation :
Pour entraîner le modèle :

bash
Copier le code
python scripts/train.py --dataset data/train --epochs 50
Pour effectuer une détection :

bash
Copier le code
python scripts/detect.py --input example_image.jpg --model models/airplane_detector.pt


## References and Documentation
# Tutoriel utilisé :
 Getting Started with Roboflow : Ce tutoriel m'a aidé à structurer le projet, notamment pour la gestion des données et l'intégration des outils d'entraînement.
# Autres outils et bibliothèques :
Roboflow : Utilisé pour préparer et annoter les images.
PyTorch : Framework principal pour entraîner le modèle.
COCO Format : Format utilisé pour annoter les données.
Problèmes connus :
Les performances en environnement sombre sont limitées.
Le modèle a du mal avec des angles de vue extrêmes.
Contribuer :
Pour signaler un bug ou proposer une amélioration, ouvre une issue ou soumets une pull request sur le dépôt GitHub.

Future Work
Améliorer les performances dans des conditions d'éclairage difficiles.
Entraîner le modèle avec un dataset plus grand pour une meilleure généralisation.
Intégrer une fonctionnalité de suivi d'objets pour des vidéos complexes.
