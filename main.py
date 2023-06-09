import cv2
import numpy as np
lower_orange = np.array([5, 50, 50])
upper_orange = np.array([15, 255, 255])

video_path = 'WhatsApp Video 2023-06-06 at 22.55.23.mp4'
video_capture = cv2.VideoCapture(video_path)

fps = video_capture.get(cv2.CAP_PROP_FPS)
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = 'output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while video_capture.isOpened():
    ret, frame = video_capture.read()

    if not ret:
        break

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 600:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            frame = cv2.circle(frame, center, radius, (0, 0, 0), 5)
            text_radius = int(radius * 1.2)
            text_center = (int(x - text_radius), int(y + text_radius))
            cv2.putText(frame, 'Circle', text_center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    output_video.write(frame)

    cv2.imshow('Processed Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
output_video.release()
cv2.destroyAllWindows()