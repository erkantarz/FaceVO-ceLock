import os
import face_recognition
import pickle

ENCODINGS_PATH = "encodings.pickle"

known_encodings = []
known_names = []

# Proje klasöründeki her klasörde gez
for folder in os.listdir():
    if os.path.isdir(folder):
        for filename in os.listdir(folder):
            if filename.startswith("photo") and filename.endswith(".jpg"):
                image_path = os.path.join(folder, filename)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if len(encodings) > 0:
                    known_encodings.append(encodings[0])
                    known_names.append(folder)

# Encode edilen verileri kaydet
with open(ENCODINGS_PATH, "wb") as f:
    data = {"encodings": known_encodings, "names": known_names}
    pickle.dump(data, f)

# Debug çıktısı
print("[✓] Tüm klasörlerdeki yüzler encode edildi ve encodings.pickle dosyası güncellendi.")
print(f"[DEBUG] Eklenen kişiler: {set(known_names)}")
