from ultralytics import YOLO
import cv2
import numpy as np

# 1. Açı Hesaplama Fonksiyonu
def aci_hesapla(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radyan = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    aci = np.abs(radyan*180.0/np.pi)
    return 360 - aci if aci > 180.0 else aci

# 2. Modeli Yükle ve Başlat
model = YOLO('yolov8n-pose.pt')
cap = cv2.VideoCapture(0)
sayac = 0
durum = "yukari"

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # YOLO tahmini (verbose=False terminali temiz tutar)
    results = model(frame, verbose=False, conf=0.5)
    
    # İskelet çizimini içeren ana görüntü
    annotated_frame = results[0].plot() 

    # --- KRİTİK GÜVENLİK KONTROLÜ (ÇÖKMEYİ ÖNLEYEN KISIM) ---
    # results[0].keypoints var mı ve içi dolu mu?
    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        
        # Noktaları alıyoruz ama listenin boş olmadığını yukarıda garanti ettik
        keypoints_data = results[0].keypoints.xy.cpu().numpy()
        
        # En az bir kişi tespit edildiyse (index 0 kontrolü)
        if keypoints_data.shape[0] > 0:
            noktalar = keypoints_data[0] # İlk tespit edilen kişi
            
            # Sağ omuz(6), dirsek(8) ve bilek(10) noktaları listede var mı?
            if len(noktalar) > 10:
                try:
                    omuz = noktalar[6]
                    dirsek = noktalar[8]
                    bilek = noktalar[10]

                    # Noktalar 0,0 değilse (gerçekten görüyorsa) hesapla
                    if omuz.any() and dirsek.any() and bilek.any():
                        aci = aci_hesapla(omuz, dirsek, bilek)

                        # SAYMA MANTIĞI (Limitleri esnettim: 140-100)
                        if aci > 140:
                            durum = "yukari"
                        if aci < 100 and durum == "yukari":
                            sayac += 1
                            durum = "asagi"

                        # Açıyı ekrana yaz
                        cv2.putText(annotated_frame, f"Aci: {int(aci)}", (50, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                except Exception:
                    pass

    # Arayüz: Siyah panel üzerine sayaç
    cv2.rectangle(annotated_frame, (0, 70), (220, 160), (0, 0, 0), -1)
    cv2.putText(annotated_frame, f"SAYAC: {sayac}", (20, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Görüntüyü Göster
    cv2.imshow("Halil AI Coach - Final", annotated_frame)
    
    # 'q' ile çıkış
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()