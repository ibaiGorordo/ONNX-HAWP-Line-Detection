import cv2
import pafy

from HAWP import HAWP

# # Initialize video
# cap = cv2.VideoCapture("input.avi")

videoUrl = 'https://youtu.be/om6s2jmDJ2c'
videoPafy = pafy.new(videoUrl)
print(videoPafy.streams)
cap = cv2.VideoCapture(videoPafy.streams[-1].url)
start_time = 10  # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * 30)

# out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (1280, 720))

# Initialize line detector
model_path = "models/hawp_512x512_float32_opt.onnx"
lineDetector = HAWP(model_path, score_threshold=0.995)

cv2.namedWindow("Detected Lines", cv2.WINDOW_NORMAL)
while cap.isOpened():

    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break

    try:
        # Read frame from the video
        ret, frame = cap.read()
        if not ret:
            break
    except Exception as e:
        print(e)
        continue

    # Perform the inference in the current frame
    lines, scores = lineDetector(frame)

    output_img = lineDetector.draw(frame)
    cv2.imshow("Detected Lines", output_img)
    # out.write(output_img)

# out.release()
