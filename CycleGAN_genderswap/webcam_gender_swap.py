import cv2
from generator_model import Generator
from utils import load_checkpoint, rescale_frame
from config import transforms
import torch
import config
import numpy as np
from classifier_model import GenderClassifier
from mtcnn import MTCNN

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
tracker = cv2.TrackerCSRT_create()

classif = GenderClassifier(n_chan=3, n_dim=(256, 256)).to(config.DEVICE)
classif.load_state_dict(torch.load("gender_classifier.pth"))
classif.eval()
gen_F = Generator(img_channels=3).to(config.DEVICE)
gen_F.eval()
load_checkpoint("256genf_batch1.pth.tar", gen_F)
gen_M = Generator(img_channels=3).to(config.DEVICE)
gen_M.eval()
load_checkpoint("256genm_batch1.pth.tar", gen_M)

tracking_face = False
face_bbox = None
female_gender = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_copy = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if not tracking_face:
        faces = face_cascade.detectMultiScale(gray, 1.1, 6)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            x, x_end, y, y_end, w, h = rescale_frame(x, y, w, h, frame)
            face_bbox = (x, y, w, h)
            face = frame[y:y_end, x:x_end]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_tensor = transforms(image=face)["image"]
            face_tensor = face_tensor.unsqueeze(0).to(config.DEVICE)
            with torch.no_grad():
                female_gender = (classif(face_tensor) > 0.5).item()

            tracker.init(frame, face_bbox)
            tracking_face = True
    else:
        success, face_bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in face_bbox]
            x = max(0, x)
            y = max(0, y)
            x_end = min(frame.shape[1], x + w)
            y_end = min(frame.shape[0], y + h)

            face = frame[y:y_end, x:x_end]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_tensor = transforms(image=face)["image"]
            face_tensor = face_tensor.unsqueeze(0).to(config.DEVICE)

            with torch.no_grad():
                generated = gen_M(face_tensor) if female_gender else gen_F(face_tensor)

            generated_np = generated.squeeze(0).permute(1, 2, 0).cpu().numpy()
            generated_np = ((generated_np*0.5+0.5) * 255).astype(np.uint8)
            generated_cv = cv2.cvtColor(generated_np, cv2.COLOR_RGB2BGR)
            generated_cv = cv2.resize(generated_cv, (w, h))

            if frame[y:y_end, x:x_end].shape == generated_cv.shape:
                frame[y:y_end, x:x_end] = generated_cv
            
            #uncomment to draw a blue rectangle if detected gender is male or red if female
            if female_gender:
                cv2.rectangle(frame, (x, y), (x_end, y_end), (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x, y), (x_end, y_end), (255, 0, 0), 2)

        else:
            tracking_face = False

    combined_frame = cv2.hconcat([frame_copy, frame])
    cv2.imshow("Original Video and Gender Swapped", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
