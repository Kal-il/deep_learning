import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.distance import cosine

def face_gauge(path_to_image: str):

    img = cv2.imread(path_to_image)

    # FaceMesh é um modelo do MediaPipe que detecta 468 pontos na face.
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

    results = face_mesh.process(img[:,:,::-1]) # converte de BGR para RGB e depois processa com o facemesh

    landmarks = results.multi_face_landmarks[0] # se tiver um rosto, obtem os pontos faciais

    face_vector = []
    for landmark in landmarks.landmark:
        face_vector.append(landmark.x)
        face_vector.append(landmark.y)
        face_vector.append(landmark.z)

    return face_vector

def euclidean_distance(vec1, vec_list):
    """
    Args:
        vec1: Vetor de medidas de um rosto
        vec_list: Lista de vetores de medidas de vários rostos
    returns:
        Retorna uma lista de distancias list(float) entre o vec1 e e os vec_list
    """
    vec1 = np.array(vec1)
    vec_list = np.array(vec_list)

    return np.linalg.norm(np.array(vec1) - np.array(vec_list), axis=1)

def euclidean_distance_img_img(vec1, vec2):
    """
    Args:
        vec1: Vetor de medidas de um rosto
        vec_list: Lista de vetores de medidas de vários rostos
    returns:
        Retorna uma lista de distancias list(float) entre o vec1 e e os vec_list
    """
    return np.linalg.norm(np.array(vec1) - np.array(vec2))

def cosine_similarity(vec1, vec_list):
    vec1 = np.array(vec1)
    vec_list = np.array(vec_list)
    similariities = [1 - cosine(vec1, vec2) for vec2 in vec_list]


    return similariities

# Exemplo: comparar um novo rosto com um rosto armazenado


if __name__ == '__main__':
    img1 = face_gauge('robert.jpg')
    img2 = face_gauge('tony_stark (Robert Downey Jr.)/tony_stark11.png')

    # measures_mesma_pessoa = []
    
    # for i in range(13):
    #     img = f'tony_stark (Robert Downey Jr.)/tony_stark{i+1}.png'
    #     measure = face_gauge(img)
    #     measures_mesma_pessoa.append(measure)

    # measures_diferent_person = []
    
    # for i in range(6):
    #     print(i)
    #     img = f'Natasha_Romanoff (Scarlett Johansson)/Natasha_Romanoff{i+1}.png'
    #     measure = face_gauge(img)
    #     measures_diferent_person.append(measure)

    # distances_same_person=[]
    # distances_same_person = euclidean_distance(img1, measures_mesma_pessoa)
    # distances_same_person_cosine = cosine_similarity(img1, measures_mesma_pessoa)

    # distances_diferent_person=[]
    # distances_diferent_person = euclidean_distance(img1, measures_diferent_person)
    # distances_diferent_person_cosine = cosine_similarity(img1, measures_diferent_person)

    # print(f'\nEuclidian Distance:\n\nDistancia mesma pessoa: {distances_same_person}\n Distância pessoa diferente: {distances_diferent_person}')
    # print(f'\Cosine Distance:\n\nDistancia mesma pessoa: {distances_same_person_cosine}\n Distância pessoa diferente: {distances_diferent_person_cosine}')
    
    dist = euclidean_distance_img_img(img1, img2)
    print(dist)
    if dist < 3.1737:  # Esse valor pode variar dependendo da precisão desejada
        print("\n\nRosto reconhecido!\n\n")
    else:
        print("\n\nRosto desconhecido.\n\n")