import cv2
import numpy as np
import onnxruntime

class YOLOv7:
    def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # Initialize model
        self.initialize_model(path)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(
            path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        # Get model info
        self.get_input_details()
        self.get_output_details()

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_shape = model_inputs[0].shape
        self.input_names = [ipt.name for ipt in model_inputs]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [opt.name for opt in model_outputs]

    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)
        outputs = self.inference(input_tensor)

        return self.process_output(outputs)

    def prepare_input(self, image):
        image = cv2.resize(image, self.input_shape[2:][::-1])
        image = image / 255.0
        image = image.transpose(2, 0, 1)

        tensor = image[np.newaxis, :, :, :].astype(np.float32)

        return tensor

    def inference(self, input_tensor):
        return self.session.run(self.output_names, {self.input_names[0]: input_tensor})

    def process_output(self, outputs):
        # Turn from (3, 1, 3, 120, 68, 9) to (3, 120, 68, 9)
        predictions = np.squeeze(outputs[0])

        # Filter out object confidence scores below threshold
        print(predictions.shape)
        obj_conf = predictions[:, 4]
        # predictions = predictions.transpose(0, 2, 1, 3)
        print(obj_conf.shape)
        predictions = predictions[obj_conf > self.conf_threshold]
        obj_conf = obj_conf[obj_conf > self.conf_threshold]

        # Multiply class confidence with bounding box confidence
        predictions[:, 5:] *= obj_conf[:, np.newaxis]

        # Get the scores
        scores = np.max(predictions[:, 5:], axis=1)

        # Filter out the objects with a low score
        predictions = predictions[scores > self.conf_threshold]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 5:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

cap = cv2.VideoCapture(0)

detector = YOLOv7("models/best.onnx", conf_thres=0.5, iou_thres=0.5)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    detector.detect_objects(frame)

    # boxes, scores, class_ids = detector.detect_objects(frame)
    # combined_img = detector.draw_detections(frame, boxes, scores, class_ids)

    # cv2.imshow("Detected Objects", cqombined_img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break