import cv2
from imread_from_url import imread_from_url

from HAWP import HAWP

# Initialize line detector
model_path = "models/hawp_512x512_float32_opt.onnx"
lineDetector = HAWP(model_path, score_threshold=0.98)

# Read image
img_url = "https://upload.wikimedia.org/wikipedia/commons/0/0d/Bedroom_Mitcham.jpg"
img = imread_from_url(img_url)

# Detect lines in the image
lines, scores = lineDetector(img)

# Draw Model Output
output_img = lineDetector.draw(img)
cv2.namedWindow("Detected Lines", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Lines", output_img)
cv2.imwrite("doc/img/output.jpg", output_img)
cv2.waitKey(0)
