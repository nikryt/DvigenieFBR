# # 9. Проверка работы всех библиотек вместе
# from facenet_pytorch import MTCNN, InceptionResnetV1
# import torch
# from PIL import Image
# import cv2
# import numpy as np
#
# # Загрузка изображения
# image_path = "test_image.jpg"
# image = Image.open(image_path)
#
# # Обнаружение лиц
# mtcnn = MTCNN(keep_all=True, device="cuda" if torch.cuda.is_available() else "cpu")
# boxes, _ = mtcnn.detect(image)
#
# # Извлечение эмбеддингов
# resnet = InceptionResnetV1(pretrained="vggface2").eval().to("cuda" if torch.cuda.is_available() else "cpu")
# faces = mtcnn(image)
# if faces is not None:
#     faces = faces.to("cuda" if torch.cuda.is_available() else "cpu")
#     embeddings = resnet(faces).detach().cpu().numpy()
#
# # Вывод результатов
# if boxes is not None:
#     print(f"Found {len(boxes)} faces!")
#     print("Embeddings shape:", embeddings.shape)
# else:
#     print("No faces found.")


# # 3. Проверка работы OpenCV

# import cv2
#
# # Проверка версии OpenCV
# print("OpenCV version:", cv2.__version__)
#
# # Проверка чтения изображения
# image = cv2.imread("test_image.jpg")
# if image is not None:
#     print("Image loaded successfully!")
# else:
#     print("Failed to load image.")
#


# # 4. Проверка работы FaceNet (facenet-pytorch)

# from facenet_pytorch import MTCNN, InceptionResnetV1
# import torch
#
# # Инициализация MTCNN и FaceNet
# mtcnn = MTCNN(keep_all=True, device="cuda" if torch.cuda.is_available() else "cpu")
# resnet = InceptionResnetV1(pretrained="vggface2").eval().to("cuda" if torch.cuda.is_available() else "cpu")
#
# # Проверка загрузки модели
# print("MTCNN and FaceNet initialized successfully!")

# # 5. Проверка работы Pillow
# from PIL import Image
#
# # Создание тестового изображения
# img = Image.new("RGB", (100, 100), color="red")
# img.save("test_image.jpg")
#
# # Проверка загрузки изображения
# loaded_img = Image.open("test_image.jpg")
# print("Image loaded successfully!")

# # 6. Проверка работы scikit-learn
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
#
# # Загрузка данных
# data = load_iris()
# X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
#
# print("Data loaded and split successfully!")

# # 7. Проверка работы matplotlib
#
# import matplotlib.pyplot as plt
#
# # Создание тестового графика
# plt.plot([1, 2, 3], [4, 5, 6])
# plt.savefig("test_plot.png")
# print("Plot saved successfully!")

# # 8. Проверка работы tqdm
#
# from tqdm import tqdm
# import time
#
# # Тест прогресс-бара
# for i in tqdm(range(10)):
#     time.sleep(0.1)
# print("Progress bar test completed!")