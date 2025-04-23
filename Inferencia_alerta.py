# visionguard_inferencia_alerta.py

from ultralytics import YOLO
from email.message import EmailMessage
from dotenv import load_dotenv
import smtplib
import os
import sys
from pathlib import Path

load_dotenv()
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_DESTINO = "jcarlos.batista@gmail.com"

# Caminho absoluto do modelo
modelo_path = str(Path.home() / "runs/classify/train/weights/best.pt")

def enviar_alerta(path_imagem):
    msg = EmailMessage()
    msg['Subject'] = '🚨 Alerta: Objeto Cortante Detectado'
    msg['From'] = EMAIL_USER
    msg['To'] = EMAIL_DESTINO
    msg.set_content("Um objeto cortante foi detectado. Imagem em anexo.")

    with open(path_imagem, 'rb') as f:
        msg.add_attachment(f.read(), maintype='image', subtype='jpeg', filename=os.path.basename(path_imagem))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls()
            s.login(EMAIL_USER, EMAIL_PASS)
            s.send_message(msg)
        print("📤 Alerta enviado com sucesso!")
    except Exception as e:
        print(f"❌ Falha ao enviar o e-mail: {e}")

def detectar_objeto(imagem_path):
    if not os.path.isfile(imagem_path):
        print(f"❌ Imagem não encontrada: {imagem_path}")
        return

    if not os.path.isfile(modelo_path):
        print(f"❌ Modelo não encontrado: {modelo_path}")
        return

    model = YOLO(modelo_path)
    results = model(imagem_path)
    pred = results[0]
    classe = pred.names[pred.probs.top1]
    score = pred.probs.top1conf.item()

    print(f"🔍 Classe detectada: {classe} | Confiança: {score:.2f}")
    if classe in ["Facas", "Tesouras", "Objetos_Cortantes"] and score > 0.85:
        enviar_alerta(imagem_path)
    else:
        print("✅ Nenhum objeto cortante detectado.")

if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else "teste/faca_sala.jpg"
    detectar_objeto(img_path)
