from ultralytics import YOLO
import cv2
import numpy as np

# Caminho do seu modelo treinado
MODEL_PATH = r"run_exp_2/exp_2_tiles/weights/best.pt"

# Carregar modelo
model = YOLO(MODEL_PATH)

def draw_obb(img, obb, color=(0, 255, 0), thickness=2):
    """
    Desenha OBB do YOLO: pts no formato (4,2) já absolutos.
    """
    pts = obb.astype(int).reshape((4, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

# Abrir webcam (0 = webcam interna)
cap = cv2.VideoCapture(0)

# Ajuste da resolução da webcam (opcional)
cap.set(3, 1280)  # largura
cap.set(4, 720)   # altura

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Fazer inferência na imagem da webcam
    results = model.predict(frame, imgsz=1024, device=0, verbose=False)

    # Pegar o único resultado
    result = results[0]

    # Se o modelo tiver OBB
    if result.obb is not None:
        classes = result.obb.cls.cpu().numpy()
        obbs = result.obb.xyxyxyxy.cpu().numpy()

        # Desenhar cada OBB
        for cls, obb in zip(classes, obbs):
            draw_obb(frame, obb)

            # Nome da classe no canto do primeiro vértice
            x, y = int(obb[0][0]), int(obb[0][1])
            cv2.putText(frame, f"{int(cls)}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Mostrar na tela
    cv2.imshow("YOLOv8-OBB Real-Time Detection", frame)

    # Sai com ESC
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
