import pygame
import os
import time
import cv2
import face_recognition
import pickle
from resemblyzer import VoiceEncoder, preprocess_wav
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

WIDTH, HEIGHT = 714, 1010
FPS = 10
PASSWORD = "kartal1903"

pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Secure Access")
font = pygame.font.SysFont("Arial", 32)
clock = pygame.time.Clock()

unlock_sound = pygame.mixer.Sound("unlock.mp3")
kapali_img = pygame.image.load("entrance.gif")
acik_img = pygame.image.load("acik.gif")

wrong_sound_path = "wrong.wav"

def draw_password_box(password, blink):
    box_rect = pygame.Rect(WIDTH // 2 - 200, HEIGHT - 180, 400, 50)
    pygame.draw.rect(screen, (0, 0, 0), box_rect, border_radius=15)
    pygame.draw.rect(screen, (255, 255, 255), box_rect, 2, border_radius=15)
    stars = "*" * len(password)
    text = font.render(stars + ("|" if blink else ""), True, (255, 255, 255))
    screen.blit(text, (box_rect.x + 20, box_rect.y + 10))

def show_password_ui():
    password = ""
    blink = True
    blink_timer = 0
    unlocked = False

    while not unlocked:
        screen.blit(pygame.transform.scale(kapali_img, (WIDTH, HEIGHT)), (0, 0))
        draw_password_box(password, blink)
        pygame.display.flip()
        clock.tick(FPS)

        blink_timer += 1
        if blink_timer % 20 == 0:
            blink = not blink

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    if password == PASSWORD:
                        unlocked = True
                    else:
                        try:
                            pygame.mixer.music.load(wrong_sound_path)
                            pygame.mixer.music.play()
                            pygame.time.delay(1500)
                        except Exception as e:
                            print(f"Ses çalma hatası: {e}")
                        password = ""
                elif event.key == pygame.K_BACKSPACE:
                    password = password[:-1]
                else:
                    if len(password) < 16:
                        password += event.unicode

    unlock_sound.play()
    screen.blit(pygame.transform.scale(acik_img, (WIDTH, HEIGHT)), (0, 0))
    pygame.display.flip()
    time.sleep(1)

def draw_overlay(surface, name):
    pygame.draw.rect(surface, (255, 255, 255), (50, 50, 1180, 620), 4)
    name_text = font.render(name.upper(), True, (255, 255, 255))
    surface.blit(name_text, (640 - name_text.get_width() // 2, 50))

def record_and_verify_voice(name, threshold=0.70):
    print("Konuşma başlıyor (5 saniye)...")
    fs = 16000
    seconds = 5
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    write("test_voice.wav", fs, recording)

    ref_path = os.path.join(name, "voice.wav")
    if not os.path.exists(ref_path):
        print("Referans sesi bulunamadı.")
        return False

    encoder = VoiceEncoder()
    ref_wav = preprocess_wav(ref_path)
    ref_embed = encoder.embed_utterance(ref_wav)

    test_wav = preprocess_wav("test_voice.wav")
    test_embed = encoder.embed_utterance(test_wav)

    sim = np.dot(ref_embed, test_embed) / (np.linalg.norm(ref_embed) * np.linalg.norm(test_embed))
    print(f"Ses benzerliği skoru: {sim:.2f}")
    return sim > threshold

def show_profile_screen(profile_path, name):
    from PIL import Image
    from PIL.ExifTags import TAGS
    try:
        img = Image.open(profile_path)
        exif = img._getexif()
        if exif:
            for tag, value in exif.items():
                decoded = TAGS.get(tag, tag)
                if decoded == "Orientation":
                    if value == 3:
                        img = img.rotate(180, expand=True)
                    elif value == 6:
                        img = img.rotate(270, expand=True)
                    elif value == 8:
                        img = img.rotate(90, expand=True)
    except Exception:
        pass

    img = img.resize((1280, 720))
    img.save("temp_display.jpg")

    loaded = pygame.image.load("temp_display.jpg")
    screen = pygame.display.set_mode((1280, 720))
    screen.blit(loaded, (0, 0))
    draw_overlay(screen, name)
    pygame.display.flip()

    if not record_and_verify_voice(name):
        print("Ses doğrulaması başarısız. Erişim reddedildi.")
        text = font.render("Voice verification failed.", True, (255, 0, 0))
        screen.blit(text, (640 - text.get_width() // 2, 610))
        pygame.display.flip()
        time.sleep(2)
        pygame.quit()
        exit()
    else:
        print("Ses doğrulama başarılı.")
        text = font.render("Voice verification is successful!", True, (0, 255, 0))
        screen.blit(text, (640 - text.get_width() // 2, 610))
        pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit(); exit()
        clock.tick(30)

def start_recognition():
    known_faces = pickle.load(open("encodings.pickle", "rb"))
    video = cv2.VideoCapture(0)
    shown = False

    while not shown:
        ret, frame = video.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            matches = face_recognition.compare_faces(known_faces["encodings"], encoding)
            if True in matches:
                matchedIdx = matches.index(True)
                name = known_faces["names"][matchedIdx]
                profile_path = os.path.join(name, "profile.jpg")
                if os.path.exists(profile_path):
                    video.release()
                    cv2.destroyAllWindows()
                    show_profile_screen(profile_path, name)
                    shown = True
                    break
            else:
                screen.fill((0, 0, 0))
                text = font.render("Stranger!", True, (255, 0, 0))
                pygame.draw.line(screen, (255, 0, 0), (600, 300), (680, 380), 8)
                pygame.draw.line(screen, (255, 0, 0), (680, 300), (600, 380), 8)
                screen.blit(text, (640 - text.get_width() // 2, 400))
                pygame.display.flip()

    video.release()
    cv2.destroyAllWindows()

def main():
    show_password_ui()
    start_recognition()

if __name__ == "__main__":
    main()
