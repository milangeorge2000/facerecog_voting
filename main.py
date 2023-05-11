import cv2
import face_recognition
import pickle
import numpy as np
from firebase import firebase

# Initialize Firebase
firebase = firebase.FirebaseApplication('https://votingmachine-d679d-default-rtdb.firebaseio.com/', None)
# Load known encodings and labels
with open('EncodeFile.p', 'rb') as f:
    encodeListKnown, studentIds = pickle.load(f)

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color to RGB color
    frame_rgb = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(frame_rgb)
    face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

    # Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face is a match for known faces
        matches = face_recognition.compare_faces(encodeListKnown, face_encoding, tolerance=0.6)
        name = "Unknown"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(encodeListKnown, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = studentIds[best_match_index]
            firebase.put('/voting', 'Verify', 1)
        else:
            firebase.put('/voting', 'Verify', 0)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with the name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
video_capture.release()
cv2.destroyAllWindows()
