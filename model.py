import os
import numpy as np
import warnings

# Gereksiz TensorFlow loglarını gizle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# --- VERİ SETİNİ YÜKLE ---
DATA_PATH = "veriseti"
actions = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
print("Veriler yükleniyor...")

for action in actions:
    files = [f for f in os.listdir(os.path.join(DATA_PATH, action)) if f.endswith('.npy')]
    for file in files:
        res = np.load(os.path.join(DATA_PATH, action, file))
        sequences.append(res)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Veriyi ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# --- MODEL YAPISI (Ezber Gücü Artırılmış) ---
model = Sequential([
    Input(shape=(30, 126)),
    LSTM(64, return_sequences=True, activation='relu'),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dropout(0.4), # Esneklik payı
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# --- EĞİTİM BAŞLIYOR ---
# Epochs: 400 (Daha uzun süre çalışma)
# Batch_size: 8 (Hassas öğrenme)
print("\nModel eğitimi başlıyor...")
model.fit(X_train, y_train, epochs=200, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Modeli kaydet
model.save('isaret_dili_modeli.h5')

# --- SONUÇ PANELİ ---
import os
os.system('cls' if os.name == 'nt' else 'clear') # Terminali temizle

print("\n" + "="*50)
print("   MODEL EĞİTİMİ BAŞARIYLA TAMAMLANDI!")
print(f"\n>>> Tanınan Toplam Kelime: {len(actions)}")
print(f">>> Kelime Listesi (Sıralı): {actions}")
print("="*50 + "\n")