import cv2
import torch
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.preprocessing import Normalizer, LabelEncoder
import joblib  # For saving/loading models

# Load saved models and encoders
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load svm model
model_svm = joblib.load(r"C:\Users\nasir\Downloads\svm_model.pkl")
in_encoder = joblib.load(r"C:\Users\nasir\Downloads\in_encoder.pkl")
out_encoder = joblib.load(r"C:\Users\nasir\Downloads\out_encoder.pkl")
all_identities = joblib.load(r"C:\Users\nasir\Downloads\all_identities.pkl")  # Dict: {label: name}

def predict_identity_from_frame(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    face = mtcnn(img)
    if face is not None:
        face = face.unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = facenet(face).cpu().numpy()
        embedding = in_encoder.transform(embedding)
        yhat_class = model_svm.predict(embedding)
        yhat_prob = model_svm.predict_proba(embedding)
        class_index = yhat_class[0]
        class_probability = yhat_prob[0, class_index] * 100
        predict_name = out_encoder.inverse_transform(yhat_class)
        return predict_name[0], class_probability
    else:
        return None, None

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access camera.")
        return

    print("Press 'q' to quit...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        identity, confidence = predict_identity_from_frame(frame)
        if identity is not None:
            label = f"{all_identities[identity]} ({confidence:.1f}%)"
            cv2.putText(frame, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Real-Time Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

