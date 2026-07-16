# Gerçek Zamanlı Türk İşaret Dili Çevirmeni (Web Tabanlı)

Bu proje; web kamerası üzerinden alınan el koordinatlarını yapay zeka ile anlık olarak analiz ederek yüksek doğruluk ve minimum gecikmeyle Türk İşaret Dili'ni metne dönüştüren web tabanlı bir erişilebilirlik sistemidir.

---

## 🚀 Proje Nasıl Çalışır?

1. **İstemci (Frontend):** Tarayıcı üzerinden web kamerasına erişir, her kareyi sunucuya gönderir ve sunucudan gelen tahmini ekranda gösterir.
2. **Sunucu (Backend):** Flask tabanlı API, gelen görüntüleri MediaPipe Hands kütüphanesi ile işleyerek el eklem noktalarını (21 nokta, x-y-z koordinatları) çıkarır.
3. **Yapay Zeka (Yapay Sinir Ağı):** Ardışık 30 kare boyunca biriken koordinat verileri, eğitilmiş LSTM modelimize gönderilir. Model, hareketin hangi kelimeye ait olduğunu tahmin eder ve sonucu tarayıcıya geri döner.

---

## 🛠️ Kurulum ve Çalıştırma Adımları

Projeyi bilgisayarınızda yerel olarak çalıştırmak için aşağıdaki adımları sırasıyla takip edin:

### 1. Gereksinimlerin Yüklenmesi
Öncelikle projenin çalışması için gerekli kütüphaneleri bilgisayarınıza yükleyin. Terminal veya Komut İstemi'ni (CMD) açarak şu komutu çalıştırın:

```bash
pip install opencv-python mediapipe numpy flask flask-cors tensorflow scikit-learn 

```

## Veri Setinin Hazırlanması
veri_toplama.py dosyasını çalıştırın.

Kamera açıldığında sırasıyla eklemek istediğiniz kelimeler (örneğin: merhaba, tesekkur_ederim ve kararlılık için bos sınıfı) için kayıtları tamamlayın.

Kaydedilen veriler otomatik olarak veriseti/ klasörü altında toplanacaktır.

## Yapay Zeka Modelinin Eğitilmesi
model_egitimi.py dosyasını çalıştırın.

Bu işlem sonucunda verileriniz eğitilecek ve ana dizinde isaret_dili_modeli.h5 dosyası oluşturulacaktır.

## Sunucunun (Backend) Başlatılması
Bash
python server.py
Sunucu varsayılan olarak http://localhost:5000 adresinde çalışacaktır.

## Web Arayüzünün (Frontend) Açılması
Projenizin arayüzünü barındıran index.html dosyasını tarayıcınızda açarak test etmeye başlayın.

📁 Proje Dosya Yapısı
Plaintext
├── veriseti/               # Toplanan koordinat (.npy) verilerinin klasörü
├── isaret_dili_modeli.h5   # Eğitilmiş yapay zeka model dosyası (gitignore ile gizlenmiştir)
├── server.py               # Tahminleri işleyen Flask sunucusu
├── model_egitimi.py        # LSTM modelini eğiten Python betiği
├── .gitignore              # GitHub'a gönderilmeyecek büyük/gereksiz dosyaların filtresi
└── README.md               # Proje tanıtım ve kullanım kılavuzu

## ⚠️ Önemli Geliştirici Notları
Boş Sınıfı (bos): Kamerada hiçbir işaret yapılmadığında sistemin hatalı tahminler üretmesini engellemek amacıyla bos isimli bir negatif örneklem sınıfı sisteme dahil edilmiştir.

Hassasiyet Dengesi (threshold): server.py içerisinde tanımlanan threshold = 0.6 (veya benzeri) değeri, modelin tahminden emin olma oranını belirler.

## İletişim
İletişime geçmek isterseniz "emiraytekin61@gmail.com" e-mail adresine mail atabilirsiniz.

### Şimdiden Teşekkür Ederiz ###