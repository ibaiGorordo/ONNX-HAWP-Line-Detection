import cv2

from HAWP import HAWP

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize line detector
model_path = "models/hawp_512x512_float32_opt.onnx"
lineDetector = HAWP(model_path, score_threshold=0.99)

cv2.namedWindow("Detected Lines", cv2.WINDOW_NORMAL)
while cap.isOpened():

    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Detect lines in the image
    lines, scores = lineDetector(frame)

    output_img = lineDetector.draw(frame)
    cv2.imshow("Detected Lines", output_img)

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
