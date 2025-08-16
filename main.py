# pipeline_yolo_depth.py
import cv2
import numpy as np
import onnxruntime as ort

# -----------------
# CONFIG
# -----------------
YOLO_MODEL_PATH = "yolov8n_fp16.onnx"
DAV2_MODEL_PATH = "depth_anything_v2_small_fp16.onnx"
IMAGE_PATH = "test_img1.jpg"

CONF_THRESH = 0.51
IOU_THRESH = 0.45
YOLO_INPUT_SIZE = 640   # YOLOv8 default input
DAV2_INPUT_SIZE = 224   # Depth Anything default input

# -----------------
# PREPROCESS
# -----------------
def preprocess_yolo(image_path, input_size=640):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (input_size, input_size))
    img_resized = img_resized.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_resized, (2, 0, 1))
    img_batch = np.expand_dims(img_transposed, axis=0)
    return img_batch.astype(np.float16), img

def preprocess_dav2(image_path, h, w):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (w, h))
    img_resized = img_resized.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_resized, (2, 0, 1))
    img_batch = np.expand_dims(img_transposed, axis=0)
    return img_batch.astype(np.float16)

# -----------------
# NMS
# -----------------
def nms(boxes, scores, iou_threshold):
    boxes_xyxy = []
    for box in boxes:
        x, y, w, h = box
        boxes_xyxy.append([x, y, x + w, y + h])
    idxs = cv2.dnn.NMSBoxes(
        bboxes=boxes_xyxy,
        scores=scores,
        score_threshold=CONF_THRESH,
        nms_threshold=iou_threshold
    )
    return idxs.flatten() if len(idxs) > 0 else []

# -----------------
# MAIN PIPELINE
# -----------------
try:
    # Load YOLO
    yolo_sess = ort.InferenceSession(YOLO_MODEL_PATH, providers=["CPUExecutionProvider"])
    yolo_input_name = yolo_sess.get_inputs()[0].name
    print(f"[YOLO] Model loaded. Input shape: {yolo_sess.get_inputs()[0].shape}")

    # Preprocess
    yolo_input, orig_img = preprocess_yolo(IMAGE_PATH, YOLO_INPUT_SIZE)
    orig_h, orig_w = orig_img.shape[:2]

    # Run YOLO inference
    yolo_output = yolo_sess.run(None, {yolo_input_name: yolo_input})[0]
    yolo_output = np.squeeze(yolo_output).reshape(84, -1).T  # (8400,84)

    # Parse YOLO detections
    boxes, scores, classes = [], [], []
    for det in yolo_output:
        x_center, y_center, w, h = det[0:4]
        class_scores = det[4:84]
        class_probs = 1 / (1 + np.exp(-class_scores))  # Sigmoid
        cls_id = np.argmax(class_probs)
        cls_conf = class_probs[cls_id]
        if cls_conf > CONF_THRESH:
            x_center = x_center * (orig_w / YOLO_INPUT_SIZE)
            y_center = y_center * (orig_h / YOLO_INPUT_SIZE)
            w = w * (orig_w / YOLO_INPUT_SIZE)
            h = h * (orig_h / YOLO_INPUT_SIZE)
            x1 = int(max(0, min(orig_w - 1, x_center - w / 2)))
            y1 = int(max(0, min(orig_h - 1, y_center - h / 2)))
            w = int(min(orig_w - x1, w))
            h = int(min(orig_h - y1, h))
            boxes.append([x1, y1, w, h])
            scores.append(float(cls_conf))
            classes.append(int(cls_id))

    indices = nms(boxes, scores, IOU_THRESH)
    output_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)

    class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                   "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                   "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                   "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                   "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                   "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                   "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                   "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
                   "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                   "scissors", "teddy bear", "hair drier", "toothbrush"]

    for i in indices:
        x, y, w, h = boxes[i]
        cls_id, conf = classes[i], scores[i]
        color = (0, 255, 0)
        cv2.rectangle(output_img, (x, y), (x + w, y + h), color, 2)
        label = f"{class_names[cls_id]}: {conf:.2f}" if cls_id < len(class_names) else f"Class{cls_id}: {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(output_img, (x, y - th - 5), (x + tw, y), color, -1)
        cv2.putText(output_img, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imwrite("YOLO_RESULT.jpg", output_img)
    print("[YOLO] Saved YOLO_RESULT.jpg")

    # -----------------
    # Depth Anything V2
    # -----------------
    dav2_sess = ort.InferenceSession(DAV2_MODEL_PATH, providers=["CPUExecutionProvider"])
    dav2_input_name = dav2_sess.get_inputs()[0].name
    dav2_input = preprocess_dav2(IMAGE_PATH, DAV2_INPUT_SIZE, DAV2_INPUT_SIZE)
    dav2_output = dav2_sess.run(None, {dav2_input_name: dav2_input})[0]

    depth_map = np.squeeze(dav2_output).astype(np.float32)
    depth_map_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_8bit = depth_map_norm.astype(np.uint8)
    depth_map_color = cv2.applyColorMap(depth_map_8bit, cv2.COLORMAP_INFERNO)

    # Resize depth map back to original image size
    depth_map_resized = cv2.resize(depth_map_color, (orig_w, orig_h))
    cv2.imwrite("DEPTH_RESULT.jpg", depth_map_resized)
    print("[DAv2] Saved DEPTH_RESULT.jpg")

except Exception as e:
    print(f"Error: {str(e)}")
