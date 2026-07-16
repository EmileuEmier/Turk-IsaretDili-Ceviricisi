import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- AYARLAR ---
DATA_PATH = "veriseti"
sequence_length = 30 

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Stil Ayarları (Neon Stil)
HAND_POINT_STYLE = mp_drawing.DrawingSpec(color=(150, 0, 0), thickness=2, circle_radius=3)
HAND_CONN_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)

def extract_landmarks(results):
    lh = np.zeros(21*3)
    rh = np.zeros(21*3)
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[i].classification[0].label
            coords = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            if label == 'Left': lh = coords
            else: rh = coords
    return np.concatenate([lh, rh])

def get_record_count(folder_name):
    """Klasördeki toplam .npy dosyası sayısını döndürür."""
    path = os.path.join(DATA_PATH, folder_name)
    if os.path.exists(path):
        return len([f for f in os.listdir(path) if f.endswith('.npy')])
    return 0

cap = cv2.VideoCapture(0)


with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    current_label = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        # El Çizimlerini Yap
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, HAND_POINT_STYLE, HAND_CONN_STYLE)
        
        # --- EKRAN BİLGİLERİ (SAĞ ÜST VE ALT) ---
        record_count = get_record_count(current_label) if current_label else 0
        
        # Sol Üst: Aktif Kelime
        cv2.putText(frame, f"KELIME: {current_label if current_label else 'SECILMEDI'}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Sağ Üst: Toplam Kayıt Sayısı
        cv2.putText(frame, f"KAYIT SAYISI: {record_count}", (380, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Alt Kısım: Tuş Rehberi
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 430), (640, 480), (0, 0, 0), -1) 
        alpha = 0.50
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


        cv2.putText(frame, "E: Kelime Gir | Q: Kayit Baslat | ESC: Cikis", (10, 465), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('Isaret Dili Veri Toplama Paneli', frame)
        
        key = cv2.waitKey(1)
        
        # 'e' tuşu: Kelime belirle
        if key == ord('e'):
            print("\n")
            current_label = input("Kaydedilecek kelime ismini girin: ").strip()
            if current_label and not os.path.exists(os.path.join(DATA_PATH, current_label)):
                os.makedirs(os.path.join(DATA_PATH, current_label))
        
        # 'q' tuşu: Kaydı başlat
        elif key == ord('q'):
            if not current_label:
                print("\n")
                print("Hata: Once bir kelime ismi belirleyin (E tusu)!")
                continue
            
            sequence = []
            time.sleep(0.5)
            print(f"Kayit basliyor: {current_label} (Kayit #{record_count + 1})")
            
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Kayıt sırasında da el çizgilerini göster
                if res.multi_hand_landmarks:
                    for hand_landmarks in res.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, HAND_POINT_STYLE, HAND_CONN_STYLE)
                
                sequence.append(extract_landmarks(res))
                
                # Ekranda geri sayım/ilerleme göster
                cv2.putText(frame, f"KAYDEDILIYOR... {frame_num}", (160, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.imshow('Isaret Dili Veri Toplama Paneli', frame)
                cv2.waitKey(1)
            
            # Kaydet
            file_path = os.path.join(DATA_PATH, current_label, f"{int(time.time())}.npy")
            np.save(file_path, sequence)
            
        elif key == 27: # ESC
            break

cap.release()
cv2.destroyAllWindows()