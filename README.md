# 🍏 Apple Leaf Disease Detection (Computer Vision)

Модель глубокого обучения (CNN) на PyTorch для автоматического распознавания заболеваний яблоневых листьев. Проект разработан как MVP (Minimum Viable Product) для сферы AgriTech.

## 🎯 Описание задачи
Нейросеть классифицирует входное изображение листа на 4 категории:
1. `Apple_scab` (Парша)
2. `Black_rot` (Черная гниль)
3. `Cedar_apple_rust` (Ржавчина)
4. `Healthy` (Здоровый лист)

**Достигнутая точность (Validation Accuracy):** ~94.8%

## 🛠 Технологический стек
* **Фреймворк:** PyTorch, TorchVision
* **Архитектура:** Custom CNN (Conv2d, BatchNorm2d, MaxPool2d, Dropout, ReLU)
* **Аппаратное ускорение:** Apple Silicon GPU (MPS) / CUDA
* **Техники:** Data Augmentation, Inverted Dropout, Batch Normalization