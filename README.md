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

![image](https://github.com/user-attachments/assets/97dd451f-4c96-43b7-9f4b-758b03a76d56)







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
![image](https://github.com/user-attachments/assets/5329432e-bd6c-4648-928a-dd9f1861af91)


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


# References and Documentation
## Tutoriel utilisé :
 Getting Started with Roboflow : Ce tutoriel m'a aidé à structurer le projet, notamment pour la gestion des données et l'intégration des outils d'entraînement.
## Autres outils et bibliothèques :
Roboflow : Utilisé pour préparer et annoter les images.
PyTorch : Framework principal pour entraîner le modèle.
COCO Format : Format utilisé pour annoter les données.
## Problèmes connus :
Les performances en environnement sombre sont limitées.
Le modèle a du mal avec des angles de vue extrêmes.
le modèle detecte tout type d'avion mais ne peut pas les différencier (j'avais deja essayer avec un premier projet mais les resultat était peu concluant)
## Point a Ameliorer : 
Améliorer les performances dans des conditions d'éclairage difficiles.
Entraîner le modèle avec un dataset plus grand pour une meilleure généralisation.
Intégrer une fonctionnalité de suivi d'objets pour des vidéos complexes.

# **Custom Airplane Detection Model**

## Project Results and Overview  
This project aims to develop a custom model for detecting airplanes in images and videos. The main goal was to create a robust model capable of identifying airplanes with precision, using a custom dataset and modern training techniques.

### **Primary Objectives :**
- Develop a supervised learning model for airplane detection to aid in recognition and identification.  
- Use a database of 300 reference images.  
- Deploy the model to detect airplanes in real time or on recorded media.  

### **Summary of Results :**
- **Accuracy Achieved :** 90.9% on a validation dataset.  
- **Performance :** Capable of detecting an airplane in less than 0.2 seconds per image.  
- **Resource Usage :** Optimized to run on configurations with a GPU (e.g., NVIDIA RTX 3060) or even on CPU with slightly longer processing times.  

### **Context :**
This project was completed as part of my final project for IMAGE UNDERSTANDING at Kyungpook National University. To achieve this, I used the OPENCV and Inference_sdk libraries, following the tutorial [Getting Started with Roboflow](https://blog.roboflow.com/getting-started-with-roboflow/), which provided a foundation for structuring the data preparation and training pipeline.  
The model was trained with a custom dataset of over 250 images, collected and annotated to meet the specific requirements of the project.  
The initial goal was to create a program that could detect a transport airplane and recognize it if it flew over a military base, using a recognition model I would train.  
To do this, I trained, using Roboflow, a model that recognizes airplanes and is trained with their AI and a model called COCO V3 (which I decided to use since YOLO V9 did not work on my computer).  
I first created a prototype model for testing, and then I developed a new version with 200 more images for training.  
I also added several data augmentations, including rotations, cropping, and brightness variations, to create the most usable images possible.  

![image](https://github.com/user-attachments/assets/2ff38dd9-61e6-4ac2-8d01-c08a8cb7c81a)  
Here, we can see the different versions and, notably, the difference between Precision and mAP.

You can find my model (and try it directly on your browser) via this link:  
https://app.roboflow.com/benoit/plane-r9j7j/models

![image](https://github.com/user-attachments/assets/97dd451f-4c96-43b7-9f4b-758b03a76d56)







## Source Code
the source code is build like this:
```bash
import cv2  // on importe OpenCV
from inference_sdk import InferenceHTTPClient 

# Initialiser le client avec l'URL et la clé API
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",  
    api_key="***************"     
)

# Définir le modèle à utiliser
model_id = "airplanes-n1kvk/1"  

# Initialiser la webcam
cap = cv2.VideoCapture(0)  

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
```

### **necessaries librairies :**
- Python 3.8+
- PyTorch >= 1.9
- Roboflow
- OpenCV
- Matplotlib


Installation :
```bash
pip install -r requirements.txt
```
Installation des librairies :
```bash
pip install openCV-python
pip install inference_sdk
```


Performance Metrics
Précision (mAP)	90.8%
Time	0.18 sec/image
memory comsomation	~2.3 GB (GPU NVIDIA)
evolution:
![image](https://github.com/user-attachments/assets/ab4bd86c-c282-4894-8533-557e6be6089b)


Installation and Usage
Installation :
clone the project :

bash
git clone https://github.com/username/custom-airplane-detector.git
cd custom-airplane-detector

bash
pip install -r requirements.txt
Configurethee dataset by following the instruction data/README.md.

Utilisation :
Pour entraîner le modèle :

bash
Copy the code 
python scripts/train.py --dataset data/train --epochs 50
Pour effectuer une détection :

bash
Copier le code
python scripts/detect.py --input example_image.jpg --model models/airplane_detector.pt


# References and Documentation
## Tutorial used :
 Getting Started with Roboflow : Ce tutoriel m'a aidé à structurer le projet, notamment pour la gestion des données et l'intégration des outils d'entraînement. this tutorial helped me tp structure the project, specially for the data utilisation and for the integration of the training tools.
## Other tools and library :
Roboflow : Used for préparation and annotation of the images.
PyTorch : Framework principal pour entraîner le modèle.
COCO Format : Format utilisé pour annoter les données.
## Problèmes connus :
Les performances en environnement sombre sont limitées.
Le modèle a du mal avec des angles de vue extrêmes.
le modèle detecte tout type d'avion mais ne peut pas les différencier (j'avais deja essayer avec un premier projet mais les resultat était peu concluant)
## Point a Ameliorer : 
Améliorer les performances dans des conditions d'éclairage difficiles.
Entraîner le modèle avec un dataset plus grand pour une meilleure généralisation.
Intégrer une fonctionnalité de suivi d'objets pour des vidéos complexes.

# **맞춤형 항공기 탐지 모델**

## Project Results and Overview
이 프로젝트는 이미지와 영상에서 항공기를 탐지하기 위한 맞춤형 모델을 개발하는 것을 목표로 합니다. 주요 목표는 정확하게 항공기를 탐지할 수 있는 강력한 모델을 제작하는 것으로, 맞춤형 데이터셋과 최신 학습 기법을 활용했습니다.

### **Objectifs principaux :**
- 항공기 탐지 및 인식을 위한 지도 학습 모델 개발.  
- 300장의 참조 이미지를 포함한 데이터베이스 활용.  
- 실시간 또는 녹화된 자료에서 항공기를 탐지할 수 있도록 모델 배포.  

### **Résumé des résultats :**
- **얻어진 정확도 :** 검증 데이터셋에서 90.9%.  
- **성능 :** 이미지 한 장당 0.2초 이내에 항공기 탐지 가능.  
- **자원 활용 :** NVIDIA RTX 3060과 같은 GPU 구성에서 최적화되었으며, CPU에서도 약간 더 긴 처리 시간으로 동작 가능.  

### **Contexte :**
이 프로젝트는 경북대학교의 IMAGE UNDERSTANDING 최종 프로젝트로 진행되었습니다. 이 프로젝트를 위해 OPENCV와 Inference_sdk 라이브러리를 활용했으며, 데이터 준비와 학습 파이프라인 구조화를 위해 [Getting Started with Roboflow](https://blog.roboflow.com/getting-started-with-roboflow/) 튜토리얼을 참고했습니다.  
이 모델은 250장이 넘는 맞춤형 이미지 데이터셋으로 학습되었으며, 항공기 탐지 및 인식을 위한 요구 사항에 맞게 수집 및 주석 처리되었습니다.  

기본 목표는 군사 기지 상공을 비행하는 수송기를 탐지하고 인식할 수 있는 프로그램을 만드는 것이었습니다. 이를 위해 Roboflow를 사용해 COCO V3 모델을 기반으로 맞춤형 학습을 진행했습니다. YOLO V9는 제 컴퓨터에서 동작하지 않아 사용하지 않았습니다.  

첫 번째 테스트 버전 이후, 추가로 200장의 이미지 데이터를 활용해 더 발전된 두 번째 모델을 개발했습니다. 또한, 회전, 자르기, 밝기 변화 등의 여러 데이터 증강 기법을 사용해 더욱 견고한 학습 데이터를 제작했습니다.  
![image](https://github.com/user-attachments/assets/2ff38dd9-61e6-4ac2-8d01-c08a8cb7c81a)  
위 이미지는 모델의 다른 버전과 정확도, mAP 비교를 보여줍니다.  

제 모델을 확인하고 브라우저에서 바로 테스트하려면 다음 링크를 클릭하세요 :  
[https://app.roboflow.com/benoit/plane-r9j7j/models](https://app.roboflow.com/benoit/plane-r9j7j/models)  
![image](https://github.com/user-attachments/assets/97dd451f-4c96-43b7-9f4b-758b03a76d56)  

## Source Code
소스 코드는 다음과 같이 구성되어 있습니다 :  
```bash
import cv2  // OpenCV 가져오기
from inference_sdk import InferenceHTTPClient  // ROBOFLOW 관리 라이브러리 가져오기

# URL 및 API 키를 사용하여 클라이언트 초기화
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",  // Roboflow 회원 API 활용
    api_key="***************"  // 개인 API 키 
)

# 사용할 모델 정의
model_id = "airplanes-n1kvk/1"  // 학습된 모델 ID

# 웹캠 초기화
cap = cv2.VideoCapture(0)  // 0은 기본 웹캠을 의미

if not cap.isOpened():  // 웹캠 접근 실패 시 처리
    print("Erreur : Impossible d'accéder à la webcam.")
    exit()

print("Appuyez sur 'q' pour quitter.")  // 프로그램 종료를 위한 키 안내

while True:
    # 웹캠에서 이미지 읽기
    ret, frame = cap.read()  
    if not ret:
        print("Erreur : Impossible de lire le flux de la webcam.")  
        break

    # 캡처한 이미지 표시
    cv2.imshow("Webcam - Appuyez sur 'q' pour quitter", frame)

    # 임시로 이미지를 저장 후 API로 전송  
    image_path = "temp_image.jpg"
    cv2.imwrite(image_path, frame)

    try:
        # Roboflow API로 이미지 전송
        with open(image_path, "rb") as image_file:  
            result = CLIENT.infer(image_file, model_id=model_id)

        # 탐지 결과 표시
        print("Résultat de la détection :")
        print(result)
    except Exception as e:
        print(f"Erreur : {e}")

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
![image](https://github.com/user-attachments/assets/d716ebb0-4f58-4fd6-ae1a-c2f113a20291)

