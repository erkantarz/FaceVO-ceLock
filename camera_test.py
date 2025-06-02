import cv2

cap = cv2.VideoCapture(0)  # Gerekirse 0 yerine 1 ya da 2 dene

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı")
        break
    cv2.imshow("Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
