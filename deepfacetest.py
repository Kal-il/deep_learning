import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from deepface import DeepFace

backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'fastmtcnn',
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yolov11s',
  'yolov11n',
  'yolov11m',
  'yunet',
  'centerface',
]



DeepFace.stream("C:/Users/kal-il/PycharmProjects/face_recognition/pics", time_threshold = 10, frame_threshold = 24, anti_spoofing = True)

result = DeepFace.verify(
  img1_path = "Natasha_Romanoff (Scarlett Johansson)/Natasha_Romanoff1.png",
  img2_path = "scarlet.jpg",
  distance_metric = "euclidean_l2",
  # model_name= "Facenet",
  detector_backend = 'retinaface',
  threshold=0.3147,
  anti_spoofing=True
)


dfs = DeepFace.find(
  img_path = "scarlet.jpg",
  model_name = 'Facenet',
  detector_backend = 'retinaface',
  db_path = "C:/Users/kal-il/PycharmProjects/face_recognition/Natasha_Romanoff (Scarlett Johansson)",
)


print(dfs)
if dfs:
    for df in dfs:
        for _, row in df.iterrows():  # Itera sobre as linhas do DataFrame
            mesma_pessoa = "sim" if row['distance'] <= row['threshold'] else "não"
            print(f"mesma pessoa: {mesma_pessoa}")
else:
    print('não é a mesma pessoa')


if result:
    
    print('\n\n')

    print(f"Threshold: {result.get('threshold')}")
    print(f"Distância calculada: {result.get('distance'):.4f}")
    print(f"Mesma pessoa? {'Sim' if result.get('verified') else 'Não'}")
    print(f"tempo de execução: {result.get('time')}")

    print('\n\n')




# Face recognition - Demo

# Face recognition requires applying face verification many times. Herein, deepface has an out-of-the-box find function
# to handle this action. It's going to look for the identity of input image in the database path and it will return list
# of pandas data frame as output. Meanwhile, facial embeddings of the facial database are stored in a pickle file to be
# searched faster in next time. Result is going to be the size of faces appearing in the source image. Besides, target 
# images in the database can have many faces as well.

