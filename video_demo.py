import cv2
import tensorflow as tf
from visualize_cv2 import model, display_instances, class_names
import sys


args = sys.argv
if len(args) < 2:
    print("Usage: python video_demo.py <video_source>")
    sys.exit(1)

video_source = args[1]

#mAP, mAR, f1_score = evaluate_model(dataset, model, config)
#print(f"mAP: {mAP}, mAR: {mAR}, F1 Score: {f1_score}")

try:
    stream = cv2.VideoCapture(video_source)
    if not stream.isOpened():
        print(f"Error: Unable to open video source '{video_source}'")
        sys.exit(1)

    

    while True:
        ret, frame = stream.read()
        if not ret:
            print("Unable to fetch frame")
            break

        results = model.detect([frame], verbose=1)
        r = results[0]
        masked_image = display_instances(frame, r['rois'], r['masks'], r['class_ids'],
                                         class_names, r['scores'])

        cv2.imshow("masked_image", masked_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    stream.release()
    cv2.destroyWindow("masked_image")
    print("Video stream ended.")

except Exception as e:
    print(f"Error occurred: {str(e)}")
    sys.exit(1)
