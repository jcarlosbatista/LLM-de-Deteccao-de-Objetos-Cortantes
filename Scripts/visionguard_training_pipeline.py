# visionguard_training_pipeline.py
# VisionGuard - Treinamento do Modelo Classificador de Objetos Cortantes com barra de progresso

import os
import random
import shutil
from ultralytics import YOLO
from tqdm import tqdm

# Caminhos iniciais
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_ORIGINAL = os.path.join(SCRIPT_DIR, "Dataset")
PASTA_TREINO = os.path.join(SCRIPT_DIR, "visionguard_dataset/train")
PASTA_VALIDACAO = os.path.join(SCRIPT_DIR, "visionguard_dataset/val")

# Etapa 1 - Criar estrutura de treino/val
CLASSES = ["Facas", "Tesouras", "Outros_Cortantes"]
RATIO_VAL = 0.2  # 20% para validação

def dividir_dataset():
    print("📁 Dividindo dataset em treino e validação...")
    total_classes = len(CLASSES)

    for idx, classe in enumerate(tqdm(CLASSES, desc="Processando classes")):
        arquivos = os.listdir(os.path.join(PASTA_ORIGINAL, classe))
        random.shuffle(arquivos)
        qtde_val = int(len(arquivos) * RATIO_VAL)

        val_set = arquivos[:qtde_val]
        train_set = arquivos[qtde_val:]

        for tipo, lista in zip(["train", "val"], [train_set, val_set]):
            destino = os.path.join(SCRIPT_DIR, "visionguard_dataset", tipo, classe)
            os.makedirs(destino, exist_ok=True)
            for arquivo in tqdm(lista, desc=f"{classe} -> {tipo}", leave=False):
                origem = os.path.join(PASTA_ORIGINAL, classe, arquivo)
                shutil.copy(origem, os.path.join(destino, arquivo))

    print("✅ Dataset dividido em treino e validação.")

# Etapa 2 - Treinamento com YOLOv8 (classificação de imagem)
def treinar_modelo():
    print("🚀 Iniciando treinamento do modelo...")
    model = YOLO('yolov8n-cls.pt')
    model.train(data='visionguard_dataset', epochs=20, imgsz=224)
    print("✅ Modelo treinado e salvo em: runs/classify/train/weights/best.pt")

# Execução do pipeline de treinamento
if __name__ == "__main__":
    dividir_dataset()
    treinar_modelo()
