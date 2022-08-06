import time
import cv2
import numpy as np
import onnxruntime


class HAWP:

    def __init__(self, path, score_threshold=0.97):
        self.threshold = score_threshold

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.update(image)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path,
                                                    providers=['CUDAExecutionProvider',
                                                               'CPUExecutionProvider'])
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def update(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        # Process output data
        self.lines, self.scores = self.process_output(outputs)

        return self.lines, self.scores

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # mean = [109.730, 103.832, 98.681]
        # std = [22.275, 22.124, 23.229]

        # input_img = ((input_img - mean) / std)
        input_img = ((input_img / 255.0 - mean) / std)
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, outputs):
        lines, scores = outputs
        lines = lines[scores > self.threshold, :]
        scores = scores[scores > self.threshold]

        lines[:, 0::2] *= self.img_width / self.input_width
        lines[:, 1::2] *= self.img_height / self.input_height
        lines = np.round(lines).astype(int)

        return lines, scores

    def draw(self, image):
        line_width = self.img_height // 200
        for line in self.lines:
            angle = np.arctan2(line[3] - line[1], line[2] - line[0])
            color = self.angle_to_color(angle)
            cv2.line(image, (line[0], line[1]),
                     (line[2], line[3]), color, line_width)
        return image

    def angle_to_color(self, angle):
        if angle < 0:
            angle = np.pi + angle

        if angle < np.pi / 3:
            return (170, 34, 226)
        elif angle < 2 * np.pi / 3:
            return (34, 226, 170)
        elif angle < np.pi:
            return (255, 72, 87)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


if __name__ == '__main__':
    from imread_from_url import imread_from_url

    model_path = "../models/hawp_512x512_float32_opt.onnx"

    # Initialize model
    lineDetector = HAWP(model_path, score_threshold=0.97)

    img = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/0/0d/Bedroom_Mitcham.jpg")

    # Perform the inference in the image
    lines, scores = lineDetector(img)

    # Draw model output
    output_img = lineDetector.draw(img)
    cv2.namedWindow("Detected Lines", cv2.WINDOW_NORMAL)
    cv2.imshow("Detected Lines", output_img)
    cv2.waitKey(0)
