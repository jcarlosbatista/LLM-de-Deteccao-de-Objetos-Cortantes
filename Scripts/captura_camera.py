import cv2
import os
from datetime import datetime
from ultralytics import YOLO
from pathlib import Path

# Caminho do modelo treinado
MODELO_PATH = str(Path.home() / "runs/classify/train2/weights/best.pt")
model = YOLO(MODELO_PATH)

# Configurações
PASTA_SAIDA = "capturas"
CONF_THRESHOLD = 0.85
OBJETOS_CORTANTES = ["Facas", "Tesouras", "Objetos_Cortantes"]
os.makedirs(PASTA_SAIDA, exist_ok=True)

# Inicializa webcam
camera = None
for idx in range(5):
    test_cam = cv2.VideoCapture(idx)
    if test_cam.isOpened():
        print(f"📷 Câmera encontrada no índice {idx}")
        camera = test_cam
        break
    test_cam.release()

if not camera or not camera.isOpened():
    print("❌ Nenhuma câmera disponível foi detectada.")
    exit()

print("🎥 VisionGuard - Pressione [ESC] para sair.")
contador = 1
captura_realizada = False  # controle para não capturar múltiplas vezes

while True:
    ret, frame = camera.read()
    if not ret:
        print("❌ Erro ao capturar imagem.")
        break

    # Convertendo de BGR (OpenCV) para RGB (YOLO)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Realiza a inferência direto no frame (numpy array)
    results = model.predict(source=frame_rgb, imgsz=224)
    pred = results[0]
    classe = pred.names[pred.probs.top1]
    score = pred.probs.top1conf.item()

    label = f"{classe} ({score:.2f})"
    cor = (0, 0, 255) if classe in OBJETOS_CORTANTES and score > CONF_THRESHOLD else (200, 200, 200)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)
    cv2.imshow("VisionGuard - Câmera ao Vivo", frame)

    # Captura apenas uma vez por detecção
    if classe in OBJETOS_CORTANTES and score > CONF_THRESHOLD:
        if not captura_realizada:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nome_arquivo = f"detectado_{classe}_{contador}_{timestamp}.jpg"
            caminho = os.path.join(PASTA_SAIDA, nome_arquivo)
            cv2.imwrite(caminho, frame)  # Salva em BGR (formato original da câmera)
            print(f"📸 Imagem capturada: {caminho}")
            contador += 1
            captura_realizada = True
    else:
        captura_realizada = False  # libera nova captura quando objeto sair da câmera

    if cv2.waitKey(1) == 27:  # ESC
        break

camera.release()
cv2.destroyAllWindows()