# LLM-de-Deteccao-de-Objetos-Cortantes


# VisionGuard: Detecção Inteligente de Objetos Cortantes com IA 🛡️🔪

Sistema de **visão computacional** para monitorar ambientes e detectar **objetos cortantes** (facas, tesouras, etc.) em tempo real usando **YOLOv8 + Python**. Quando um objeto perigoso é identificado, o sistema **envia um alerta automático por e-mail**.

![VisionGuard Overview](link_da_imagem_arquitetura.png)



VisionGuard: Detecção Inteligente de Objetos Cortantes com IA

Bem-vindo ao VisionGuard, um sistema completo de detecção de objetos cortantes (facas, tesouras, etc.) usando Inteligência Artificial com YOLOv8 e Python. Este projeto foi desenvolvido para monitorar ambientes em tempo real e enviar alertas automáticos por e-mail sempre que objetos potencialmente perigosos forem detectados.



## 🚀 Funcionalidades

- 🎯 Treinamento supervisionado com YOLOv8 (classificação).
- 🔍 Validação manual de imagens específicas.
- 📂 Monitoramento automatizado de pastas com imagens.
- 🎥 Captura de imagens via webcam em tempo real.
- 📧 Envio automático de e-mails com alertas.



🧱 Estrutura do Projeto

VisionGuard
<pre>
VisionGuard
- Dataset                            # Dataset de treinamento (Facas, Tesouras, Objetos_Cortantes)
- capturas                           # Pasta monitorada com imagens capturadas
  - UrgenteAlerta                    # Imagens com objetos cortantes detectados
  - Analisado                        # Imagens analisadas sem risco detectado 
- runs                               # Diretório de modelos treinados (YOLOv8)
- visionguard_training_pipeline.py   # Script de treinamento
- inferencia_alerta.py               # Script de validação manual
- monitoramento_alerta_email.py      # Script de monitoramento e alerta
- captura_camera.py                  # Script de captura de imagens via webcam
</pre>




## **Pré-requisitos**

Python 3.8+

Bibliotecas Python:

``` 
ultralytics (YOLOv8)
opencv-python
python-dotenv
tqdm
pillow
```

Crie um arquivo .env com suas credenciais de e-mail
```
EMAIL_USER=seuemail@gmail.com
EMAIL_PASS=sua_senha_ou_app_password
```

🏗️ Etapas do Projeto

1️⃣ Treinamento do Modelo

Arquivo: visionguard_training_pipeline.py

Divide o dataset em 80% treino e 20% validação.
Treina o modelo YOLOv8-cls por 20 epochs para classificar imagens em:

- Facas
- Tesouras
- Objetos_Cortantes

Comando:
```
python visionguard_training_pipeline.py
```


2️⃣ Validação Manual de Imagens

Arquivo: inferencia_alerta.py
Permite validar manualmente uma imagem e enviar um alerta por e-mail se um objeto cortante for detectado.

Comando:
```
python inferencia_alerta.py caminho/para/imagem.jpg
```

3️⃣ Monitoramento Automático de Pasta

Arquivo: monitoramento_alerta_email.py

Monitora continuamente a pasta capturas.
Sempre que uma nova imagem aparecer:
- Classifica a imagem.
- Se detectar objeto cortante com confiança > 85%:
  - Envia e-mail de alerta.
  - Move a imagem para UrgenteAlerta.
  - Caso contrário, move para Analisado.

Comando:
```
python monitoramento_alerta_email.py
```

4️⃣ Captura de Imagens em Tempo Real

Arquivo: captura_camera.py
- Detecta a webcam disponível.
- Exibe o feed ao vivo.
- Pressione barra de espaço [ESPAÇO] para capturar uma imagem.
- A imagem é salva na pasta capturas/.

Comando:
```
python captura_camera.py
```

📬 Alertas por E-mail

O envio de e-mails usa o SMTP do Gmail. Para funcionar:
- Ative o 2FA na sua conta Google.
- Gere uma senha de app.
- Configure no .env com suas credenciais.
