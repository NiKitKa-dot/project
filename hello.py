

# Создание структуры папок
import os
os.makedirs('/content/drive/MyDrive/dataset/train/images', exist_ok=True)
os.makedirs('/content/drive/MyDrive/dataset/train/labels', exist_ok=True)
os.makedirs('/content/drive/MyDrive/dataset/val/images', exist_ok=True)
os.makedirs('/content/drive/MyDrive/dataset/val/labels', exist_ok=True)
os.makedirs('/content/drive/MyDrive/weights', exist_ok=True)
os.makedirs('/content/drive/MyDrive/results', exist_ok=True)

# Конфигурационный файл dataset.yaml
yaml_content = """
path: /content/drive/MyDrive/dataset
train: train/images
val: val/images
test: val/images

names:
  0: person
"""

with open('/content/drive/MyDrive/dataset/dataset.yaml', 'w') as f:
    f.write(yaml_content.strip())

# Импорт библиотек
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import seaborn as sns
from datetime import datetime

# Инициализация модели
model = YOLO('/content/drive/MyDrive/yolov11m.pt')

# Конфигурация обучения для малого датасета
config = {
    'data': '/content/drive/MyDrive/dataset/dataset.yaml',
    'epochs': 51,
    'imgsz': 1280,
    'batch': 5,  # Уменьшенный batch size для 1000 изображений
    'save_period': 3,
    'project': '/content/drive/MyDrive/weights',
    'name': f'yolov11m_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
    'optimizer': 'AdamW',
    'lr0': 0.001,
    'cos_lr': True,
    'weight_decay': 0.0005,
    'fliplr': 0.2,
    'mosaic': 0.8,
    'mixup': 0.2,
    'close_mosaic': 15,
    'label_smoothing': 0.1,
    'dropout': 0.2,
    'patience': 10,
    'box': 0.7,  # Увеличенный вес для box loss
    'cls': 0.3   # Уменьшенный вес для class loss (только один класс)
}

# Обучение модели
results = model.train(**config)

# Функция для визуализации метрик
def plot_metrics(results_path, save_dir):
    results_csv = pd.read_csv(f'{results_path}/results.csv')

    plt.figure(figsize=(20, 15))
    sns.set_style("whitegrid")

    # Графики точности
    plt.subplot(3, 2, 1)
    plt.plot(results_csv['epoch'], results_csv['metrics/precision(B)'], label='Precision', color='blue')
    plt.title('Precision (Точность)', fontsize=12)
    plt.xlabel('Эпоха')
    plt.ylabel('Значение')
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(results_csv['epoch'], results_csv['metrics/recall(B)'], label='Recall', color='green')
    plt.title('Recall (Полнота)', fontsize=12)
    plt.xlabel('Эпоха')
    plt.ylabel('Значение')
    plt.legend()

    # Графики mAP
    plt.subplot(3, 2, 3)
    plt.plot(results_csv['epoch'], results_csv['metrics/mAP50(B)'], label='mAP@0.5', color='red')
    plt.title('mAP@0.5', fontsize=12)
    plt.xlabel('Эпоха')
    plt.ylabel('Значение')
    best_mAP50 = results_csv['metrics/mAP50(B)'].max()
    best_epoch = results_csv['metrics/mAP50(B)'].idxmax()
    plt.annotate(f'Лучшее: {best_mAP50:.3f} (эпоха {best_epoch+1})',
                 xy=(best_epoch, best_mAP50), xytext=(10, 10),
                 textcoords='offset points', arrowprops=dict(arrowstyle='->'))
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(results_csv['epoch'], results_csv['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', color='purple')
    plt.title('mAP@0.5:0.95', fontsize=12)
    plt.xlabel('Эпоха')
    plt.ylabel('Значение')
    best_mAP = results_csv['metrics/mAP50-95(B)'].max()
    best_epoch = results_csv['metrics/mAP50-95(B)'].idxmax()
    plt.annotate(f'Лучшее: {best_mAP:.3f} (эпоха {best_epoch+1})',
                 xy=(best_epoch, best_mAP), xytext=(10, 10),
                 textcoords='offset points', arrowprops=dict(arrowstyle='->'))
    plt.legend()

    # Графики потерь
    plt.subplot(3, 2, 5)
    plt.plot(results_csv['epoch'], results_csv['train/box_loss'], label='Train Box Loss', color='orange')
    plt.plot(results_csv['epoch'], results_csv['val/box_loss'], label='Val Box Loss', color='cyan')
    plt.title('Box Loss', fontsize=12)
    plt.xlabel('Эпоха')
    plt.ylabel('Значение')
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(results_csv['epoch'], results_csv['train/cls_loss'], label='Train Class Loss', color='orange')
    plt.plot(results_csv['epoch'], results_csv['val/cls_loss'], label='Val Class Loss', color='cyan')
    plt.title('Class Loss', fontsize=12)
    plt.xlabel('Эпоха')
    plt.ylabel('Значение')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{save_dir}/metrics_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

# Визуализация метрик
plot_metrics(f"{config['project']}/{config['name']}", '/content/drive/MyDrive/results')

# Функция для генерации примеров детекции
def save_detection_examples(model, test_images_dir, save_dir, num_examples=3):
    image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:num_examples]

    for img_file in image_files:
        img_path = os.path.join(test_images_dir, img_file)
        results = model.predict(img_path, conf=0.5)

        res_plotted = results[0].plot(line_width=2, font_size=10)
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.savefig(f'{save_dir}/detection_{img_file}', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

# Сохранение примеров детекции
test_images_dir = '/content/drive/MyDrive/dataset/val/images'
save_detection_examples(model, test_images_dir, '/content/drive/MyDrive/results')

# Экспорт модели в ONNX
onnx_path = '/content/drive/MyDrive/results/model.onnx'
model.export(format='onnx', dynamic=True, imgsz=640, opset=12, simplify=True)

# Функция для генерации отчета
def generate_report(results_path, save_path):
    results_csv = pd.read_csv(f'{results_path}/results.csv')
    best_mAP50 = results_csv['metrics/mAP50(B)'].max()
    best_epoch = results_csv['metrics/mAP50(B)'].idxmax() + 1

    report = f"""
ДЕТАЛЬНЫЙ ОТЧЁТ О РЕЗУЛЬТАТАХ ОБУЧЕНИЯ
Дата: {datetime.now().strftime("%Y-%m-%d %H:%M")}

=== ОБЩАЯ ИНФОРМАЦИЯ ===
Модель: YOLOv11m
Эпох обучения: {EPOCHS}
Размер датасета: 1000 изображений
Лучшая mAP@0.5: {best_mAP50:.4f} на эпохе {best_epoch}

=== ОЦЕНКА МЕТРИК ===
Ключевые метрики на лучшей эпохе:
- Precision: {results_csv.at[best_epoch-1, 'metrics/precision(B)']:.4f}
- Recall: {results_csv.at[best_epoch-1, 'metrics/recall(B)']:.4f}
- F1-Score: {results_csv.at[best_epoch-1, 'metrics/f1(B)']:.4f}
- mAP@0.5: {results_csv.at[best_epoch-1, 'metrics/mAP50(B)']:.4f}
- mAP@0.5:0.95: {results_csv.at[best_epoch-1, 'metrics/mAP50-95(B)']:.4f}

=== ИНТЕРПРЕТАЦИЯ МЕТРИК ===
1. Precision (Точность): Доля корректно обнаруженных людей среди всех обнаружений.
   - >0.85: Отлично | 0.7-0.85: Хорошо | <0.7: Требует улучшения

2. Recall (Полнота): Доля реальных людей, которые были обнаружены.
   - >0.8: Отлично | 0.6-0.8: Хорошо | <0.6: Требует улучшения

3. F1-Score: Баланс между Precision и Recall.
   - >0.8: Отлично | 0.6-0.8: Хорошо | <0.6: Низкое качество

4. mAP@0.5: Средняя точность при IoU=0.5.
   - >0.75: Отлично | 0.5-0.75: Хорошо | <0.5: Низкое качество

5. mAP@0.5:0.95: Средняя точность при разных порогах IoU (более строгая метрика).
   - >0.4: Отлично | 0.25-0.4: Хорошо | <0.25: Требует доработки

=== ОЦЕНКА КАЧЕСТВА ===
"""

    # Оценка качества
    if best_mAP50 > 0.75:
        report += "ОЦЕНКА: ОТЛИЧНО\n- Модель показывает превосходные результаты для сложных условий\n- Готова к промышленному использованию"
    elif best_mAP50 > 0.6:
        report += "ОЦЕНКА: ХОРОШО\n- Модель пригодна для использования, но требует мониторинга\n- Возможны ошибки на маленьких объектах"
    elif best_mAP50 > 0.45:
        report += "ОЦЕНКА: УДОВЛЕТВОРИТЕЛЬНО\n- Требуется дополнительная настройка\n- Необходимо улучшение для сложных случаев"
    else:
        report += "ОЦЕНКА: ПЛОХО\n- Необходимо серьезное улучшение модели\n- Проверьте качество разметки данных"

    report += "\n\n=== РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ ==="

    # Персонализированные рекомендации
    precision = results_csv.at[best_epoch-1, 'metrics/precision(B)']
    recall = results_csv.at[best_epoch-1, 'metrics/recall(B)']

    if precision < 0.7:
        report += "\n- Увеличить порог confidence для уменьшения ложных срабатываний"
        report += "\n- Добавить больше негативных примеров (кадры без людей)"

    if recall < 0.6:
        report += "\n- Увеличить аугментацию (особенно масштабирование)"
        report += "\n- Добавить примеры маленьких объектов (люди вдалеке)"

    if results_csv.at[best_epoch-1, 'metrics/mAP50-95(B)'] < 0.3:
        report += "\n- Увеличить разрешение изображений"
        report += "\n- Использовать более агрессивную mosaic аугментацию"

    report += "\n- Рассмотреть увеличение датасета до 3000+ изображений"
    report += "\n- Настроить anchor boxes под специфику БПЛА"
    report += "\n- Экспериментировать с разными оптимизаторами"

    # Сохранение отчета
    with open(f'{save_path}/report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    return report

# Генерация и сохранение отчета
report = generate_report(f"{config['project']}/{config['name']}", '/content/drive/MyDrive/results')
print("="*80)
print(report)
print("="*80)
print("Обучение завершено! Результаты сохранены в Google Drive")
