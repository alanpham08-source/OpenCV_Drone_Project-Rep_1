from inference import get_model
import supervision as sv
import cv2
from datetime import datetime

# Load model
model = get_model(
    model_id="Your Model ID",#-->  Replace with your Model ID
    api_key="Your ROBOFLOW API"#--> Replace with your API key
)

annotator = sv.BoxAnnotator()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    result = model.infer(frame)[0]

    # Convert to Supervision detections
    detections = sv.Detections.from_inference(result)

    # Annotate bounding boxes
    annotated = annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    # Get current datetime string
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Overlay labels + current datetime near each detected object
    for pred in result.predictions:
        x = int(pred.x)
        y = int(pred.y)
        w = int(pred.width)
        h = int(pred.height)

        cls = pred.class_name
        conf = int(pred.confidence * 100)

        label_text = f"{cls} – {conf}%"
        datetime_text = current_datetime

        # Text position: above the top-left corner of the box
        text_x = x - w // 2
        text_y = y - h // 2 - 10
        text_y = max(text_y, 20)  # prevent text off-screen

        # Draw class/confidence label
        cv2.putText(
            annotated,
            label_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Draw current date/time just below the label text
        cv2.putText(
            annotated,
            datetime_text,
            (text_x, text_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA
        )

    # Show frame
    cv2.imshow("Roboflow Live Stream", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

