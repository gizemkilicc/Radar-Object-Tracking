import cv2
from ultralytics import YOLO

# YOLOv8 modelini yükle
model = YOLO("yolov8s.pt")

# Kamerayı aç
cap = cv2.VideoCapture(0)

# ROI (dikdörtgen alan) koordinatları
x1, y1, x2, y2 = 200, 100, 800, 500
roi_w = x2 - x1
roi_h = y2 - y1

# Sayım için değişkenler
count = 0
counted_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Takip + tahmin
    results = model.track(
        source=frame,
        conf=0.6,
        iou=0.5,
        persist=True,
        tracker="bytetrack.yaml"
    )

    annotated_frame = results[0].plot()

    # ROI çiz
    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"ROI Alan ({roi_w}x{roi_h})", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, obj_id, cls in zip(boxes, ids, classes):
            bx1, by1, bx2, by2 = box
            cx, cy = int((bx1 + bx2) / 2), int((by1 + by2) / 2)
            w, h = int(bx2 - bx1), int(by2 - by1)  # nesne boyutları

            # ROI içinde mi?
            if x1 < cx < x2 and y1 < cy < y2:
                if obj_id not in counted_ids:
                    counted_ids.add(obj_id)
                    count += 1

            # ID, sınıf ve boyut etiketi
            label = f"ID {obj_id} - Class {cls} ({w}x{h})"
            cv2.putText(annotated_frame, label, (int(bx1), int(by1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.circle(annotated_frame, (cx, cy), 4, (0, 0, 255), -1)

    # Toplam sayıyı ekranda göster
    cv2.putText(annotated_frame, f"ROI Count: {count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Pencereyi büyüt
    cv2.namedWindow("YOLOv8 + ByteTrack", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv8 + ByteTrack", 1280, 720)

    cv2.imshow("YOLOv8 + ByteTrack", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()



import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time

# Model yükle
model = YOLO("yolov8s.pt")

# Kamera aç
cap = cv2.VideoCapture(0)

# Sayaç ve grafik için listeler
unique_ids = set()
cumulative_counts = []
active_counts = []
fps_list = []
frame_ids = []
id_history = {}   # ID -> hangi framelerde göründü

frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # Takip başlat
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")

    # ID'leri al
    ids_in_frame = []
    if results[0].boxes.id is not None:
        ids_in_frame = results[0].boxes.id.int().cpu().tolist()

    # Eşsiz ID’leri güncelle
    for i in ids_in_frame:
        unique_ids.add(i)
        if i not in id_history:
            id_history[i] = []
        id_history[i].append(frame_num)

    # Sayaçlar
    frame_num += 1
    frame_ids.append(frame_num)
    cumulative_counts.append(len(unique_ids))  # toplam kişi
    active_counts.append(len(ids_in_frame))    # o anda ekranda olan kişi

    # FPS hesapla
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    fps_list.append(fps)

    # Görselleştirme
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 + ByteTrack", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# =======================
# GRAFİKLERİ ÇİZ
# =======================
plt.figure(figsize=(14, 10))

# 1. Toplam Sayım Grafiği
plt.subplot(4, 1, 1)
plt.plot(frame_ids, cumulative_counts, label="Toplam Sayım", color="blue")
plt.xlabel("Frame")
plt.ylabel("Kişi Sayısı")
plt.title("Zamanla Toplam Kişi Sayısı")
plt.legend()

# 2. Anlık Nesne Sayısı Grafiği
plt.subplot(4, 1, 2)
plt.plot(frame_ids, active_counts, label="Anlık Kişi", color="green")
plt.xlabel("Frame")
plt.ylabel("Kişi Sayısı")
plt.title("Her Karede Görülen Kişi Sayısı")
plt.legend()

# 3. FPS Grafiği
plt.subplot(4, 1, 3)
plt.plot(frame_ids, fps_list, label="FPS", color="red")
plt.xlabel("Frame")
plt.ylabel("FPS")
plt.title("İşleme Hızı (FPS)")
plt.legend()

# 4. ID Takip Grafiği
plt.subplot(4, 1, 4)
for id_, frames in id_history.items():
    plt.scatter(frames, [id_] * len(frames), s=10, label=f"ID {id_}")
plt.xlabel("Frame")
plt.ylabel("ID")
plt.title("ID Takip Grafiği (Hangi ID hangi framelerde göründü)")
plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
plt.tight_layout()
plt.show()


import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time

# Model yükle
model = YOLO("yolov8s.pt")

# Kamera aç
cap = cv2.VideoCapture(0)

# ROI alanı (örnek: ortada bir dikdörtgen)
roi_x1, roi_y1, roi_x2, roi_y2 = 200, 100, 500, 400

# Sayaç ve grafik için listeler
unique_ids = set()
cumulative_counts = []
active_counts = []
fps_list = []
frame_ids = []
id_history = {}   # ID -> hangi framelerde göründü
widths = []       # ROI içindeki nesne genişlikleri
heights = []      # ROI içindeki nesne yükseklikleri

frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # Takip başlat
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")

    # ROI’yi çiz
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)

    # ID'leri al
    ids_in_frame = []
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().tolist()
        cls = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, cls_id in zip(boxes, ids, cls):
            x1, y1, x2, y2 = box.astype(int)
            ids_in_frame.append(track_id)

            # ROI içine giriyorsa boyut kaydet
            if roi_x1 < x1 < roi_x2 and roi_y1 < y1 < roi_y2:
                w = x2 - x1
                h = y2 - y1
                widths.append(w)
                heights.append(h)

            # ID takibi için
            if track_id not in id_history:
                id_history[track_id] = []
            id_history[track_id].append(frame_num)

            # Kutuları çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Eşsiz ID’leri güncelle
    for i in ids_in_frame:
        unique_ids.add(i)

    # Sayaçlar
    frame_num += 1
    frame_ids.append(frame_num)
    cumulative_counts.append(len(unique_ids))  # toplam kişi
    active_counts.append(len(ids_in_frame))    # o anda ekranda olan kişi

    # FPS hesapla
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    fps_list.append(fps)

    # Görselleştirme
    cv2.imshow("YOLOv8 + ByteTrack", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# =======================
# GRAFİKLERİ ÇİZ
# =======================
plt.figure(figsize=(14, 12))

# 1. Toplam Sayım
plt.subplot(4, 1, 1)
plt.plot(frame_ids, cumulative_counts, label="Toplam Sayım", color="blue")
plt.xlabel("Frame")
plt.ylabel("Kişi Sayısı")
plt.title("Zamanla Toplam Kişi Sayısı")
plt.legend()

# 2. Anlık Sayım
plt.subplot(4, 1, 2)
plt.plot(frame_ids, active_counts, label="Anlık Kişi", color="green")
plt.xlabel("Frame")
plt.ylabel("Kişi Sayısı")
plt.title("Her Karede Görülen Kişi Sayısı")
plt.legend()

# 3. FPS
plt.subplot(4, 1, 3)
plt.plot(frame_ids, fps_list, label="FPS", color="red")
plt.xlabel("Frame")
plt.ylabel("FPS")
plt.title("İşleme Hızı (FPS)")
plt.legend()

# 4. Dimension Histogramı (ROI içindeki nesneler)
plt.subplot(4, 1, 4)
plt.hist(widths, bins=10, alpha=0.5, label="Genişlik (px)")
plt.hist(heights, bins=10, alpha=0.5, label="Yükseklik (px)")
plt.xlabel("Piksel")
plt.ylabel("Adet")
plt.title("ROI İçindeki Nesnelerin Boyut Dağılımı")
plt.legend()

plt.tight_layout()
plt.show()

# =======================
# F1 Score – Confidence Threshold Grafiği
# =======================
import numpy as np

# Confidence threshold değerleri (0.1–0.9 arası)
thresholds = np.linspace(0.1, 0.9, 9)

# Dummy F1 skorları (örnek: orta eşikte daha yüksek)
f1_scores = [0.55, 0.62, 0.70, 0.78, 0.82, 0.79, 0.74, 0.68, 0.60]

plt.figure(figsize=(6,4))
plt.plot(thresholds, f1_scores, marker='o', color='purple', label="F1 Score")
plt.title("F1 Score vs Confidence Threshold")
plt.xlabel("Confidence Threshold")
plt.ylabel("F1 Score")
plt.ylim(0.5, 1.0)
plt.grid(False)  # Arkada kare olmasın
plt.legend()
plt.show()


