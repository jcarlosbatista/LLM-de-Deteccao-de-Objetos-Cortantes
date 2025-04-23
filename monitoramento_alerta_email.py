# monitoramento_alerta_email.py
# Monitoramento de pasta com inferência, barra de progresso e envio de alerta por e-mail (com .env seguro)

import os
import time
import shutil
from ultralytics import YOLO
from email.message import EmailMessage
import smtplib
from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path
from PIL import Image

def imagem_valida(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

# Carrega variáveis do .env
load_dotenv()
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_DESTINO = "jcarlos.batista@gmail.com"

# Caminho absoluto do modelo
MODELO_PATH = str(Path.home() / "runs/classify/train/weights/best.pt")

PASTA_MONITORADA = "capturas"
PASTA_ALERTA = os.path.join(PASTA_MONITORADA, "UrgenteAlerta")
PASTA_ANALISADO = os.path.join(PASTA_MONITORADA, "Analisado")
EXTENSOES_VALIDAS = [".jpg", ".jpeg", ".png"]
THRESHOLD_CONF = 0.85
OBJETOS_PERIGOSOS = ["Facas", "Tesouras", "Objetos_Cortantes"]

# Cria pastas se não existirem
os.makedirs(PASTA_ALERTA, exist_ok=True)
os.makedirs(PASTA_ANALISADO, exist_ok=True)

# Envio de alerta por e-mail
def enviar_alerta(path_imagem):
    msg = EmailMessage()
    msg['Subject'] = '🚨 VisionGuard: Objeto Cortante Detectado'
    msg['From'] = EMAIL_USER
    msg['To'] = EMAIL_DESTINO
    msg.set_content("Um objeto cortante foi identificado. Veja imagem anexa.")

    with open(path_imagem, 'rb') as f:
        msg.add_attachment(f.read(), maintype='image', subtype='jpeg', filename=os.path.basename(path_imagem))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls()
            s.login(EMAIL_USER, EMAIL_PASS)
            s.send_message(msg)
        print(f"📤 Alerta enviado para {EMAIL_DESTINO}")
        return True
    except Exception as e:
        print(f"❌ Erro ao enviar alerta: {e}")
        return False

# Inferência individual
def processar_imagem(imagem_path, model):
    results = model(imagem_path)
    pred = results[0]
    classe = pred.names[pred.probs.top1]
    score = pred.probs.top1conf.item()

    print(f"🔍 {imagem_path} | Classe: {classe} | Confiança: {score:.2f}")
    if classe in OBJETOS_PERIGOSOS and score > THRESHOLD_CONF:
        if enviar_alerta(imagem_path):
            destino = os.path.join(PASTA_ALERTA, os.path.basename(imagem_path))
            shutil.move(imagem_path, destino)
            print(f"📂 Movido para: {destino}")
    else:
        destino = os.path.join(PASTA_ANALISADO, os.path.basename(imagem_path))
        shutil.move(imagem_path, destino)
        print(f"📂 Movido para: {destino}")

# Monitoramento com barra de progresso
def monitorar_pasta(pasta):
    try:
        model = YOLO(MODELO_PATH)
    except FileNotFoundError:
        print(f"❌ Modelo não encontrado em {MODELO_PATH}. Execute o treinamento antes.")
        return

    print(f"👁️ Monitorando a pasta: {pasta}")
    processados = set()

    while True:
        arquivos = [f for f in os.listdir(pasta) if f.endswith(tuple(EXTENSOES_VALIDAS))]
        novos = [f for f in arquivos if f not in processados]

        for arquivo in tqdm(novos, desc="🧠 Analisando imagens novas"):
            caminho = os.path.join(pasta, arquivo)
            if imagem_valida(caminho):
                processar_imagem(caminho, model)
                processados.add(arquivo)
            else:
                print(f"⚠️ Imagem inválida ou corrompida: {caminho}")

        time.sleep(5)

if __name__ == "__main__":
    monitorar_pasta(PASTA_MONITORADA)