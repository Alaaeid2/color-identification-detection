import cv2
import numpy as np
import time

# ---------------------------- COLOR RANGES ---------------------------- #
COLOR_RANGES = {
    'Red': {
        'lower': np.array([136, 87, 111], np.uint8),
        'upper': np.array([180, 255, 255], np.uint8),
        'color': (0, 0, 255)
    },
    'Yellow': {
        'lower': np.array([26, 43, 46], np.uint8),
        'upper': np.array([34, 255, 255], np.uint8),
        'color': (0, 255, 255)
    },
    'Green': {
        'lower': np.array([25, 52, 72], np.uint8),
        'upper': np.array([102, 255, 255], np.uint8),
        'color': (0, 255, 0)
    },
    'Blue': {
        'lower': np.array([94, 80, 2], np.uint8),
        'upper': np.array([120, 255, 255], np.uint8),
        'color': (255, 0, 0)
    },
    'White': {
        'lower': np.array([0, 0, 200], np.uint8),
        'upper': np.array([180, 25, 255], np.uint8),
        'color': (255, 255, 255)
    },
    'Black': {
        'lower': np.array([0, 0, 0], np.uint8),
        'upper': np.array([180, 255, 30], np.uint8),
        'color': (0, 0, 0)
    },
    'Gray': {
        'lower': np.array([0, 0, 50], np.uint8),
        'upper': np.array([180, 255, 200], np.uint8),
        'color': (128, 128, 128)
    },
    'Orange': {
        'lower': np.array([5, 150, 150], np.uint8),
        'upper': np.array([15, 255, 255], np.uint8),
        'color': (0, 165, 255)
    },
    'Pink': {
        'lower': np.array([145, 100, 100], np.uint8),
        'upper': np.array([165, 255, 255], np.uint8),
        'color': (203, 192, 255)
    },
    'Purple': {
        'lower': np.array([129, 50, 70], np.uint8),
        'upper': np.array([158, 255, 255], np.uint8),
        'color': (128, 0, 128)
    }
}

KERNEL = np.ones((5, 5), "uint8")


def get_color_mask(hsv_frame, color_range):
    mask = cv2.inRange(hsv_frame, color_range['lower'], color_range['upper'])
    
    # Clean noise
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Morphological Opening: remove small objects
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    
    # Dilation: to make the detected object area bigger
    mask = cv2.dilate(mask, KERNEL, iterations=1)
    
    return mask
def draw_contours(frame, mask, label, box_color, area_threshold=1000):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > area_threshold:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(frame, f"{label} Colour", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)


# ---------------------------- MAIN LOOP ---------------------------- #
def start_color_detection():
    webcam = cv2.VideoCapture(0)

    if not webcam.isOpened():
        print("Error: Webcam not accessible.")
        return

    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for color_name, color_data in COLOR_RANGES.items():
            mask = get_color_mask(hsv_frame, color_data)
            draw_contours(frame, mask, color_name, color_data['color'])

        cv2.imshow("Color Detection", frame)
        time.sleep(0.01)  # Adjust if needed, not necessary on all setups

        # Press 'q' to exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()


# ---------------------------- ENTRY POINT ---------------------------- #
if __name__ == "__main__":
    start_color_detection()
