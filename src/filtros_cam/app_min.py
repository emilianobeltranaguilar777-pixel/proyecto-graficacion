import cv2

def run(camera_index: int = 0):
    cap = cv2.VideoCapture(camera_index)


    if not cap.isOpened():
        print(f"No pude abrir la cámara (índice {camera_index}). "
              f"Si usas DroidCam prueba con 2: run(2)")
        return

    print("Controles: [q] salir")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("No pude leer frames. ¿Cámara desconectada?")
            break

        cv2.imshow("Camara - Minimal", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
2
if __name__ == "__main__":
    # Cambia a 2 si usas DroidCam en /dev/video2
    run(camera_index=0)
