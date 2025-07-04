import os
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms
import torch

# Paths
input_base = r"C:\Users\nasir\OneDrive\Desktop\face_dataset"
output_base = r"C:\Users\nasir\OneDrive\Desktop\facedata_croped"
os.makedirs(output_base, exist_ok=True)

# Initialize MTCNN for face detection
mtcnn = MTCNN(
    margin=10,
    image_size=160,
    select_largest=True,
    post_process=True,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

def compress_image(img, output_path, max_size_kb=50):
    quality = 85
    img_format = 'JPEG'
    while True:
        img.save(output_path, format=img_format, quality=quality, optimize=True)
        size_kb = os.path.getsize(output_path) / 1024
        if size_kb <= max_size_kb or quality <= 10:
            break
        quality -= 5
    return size_kb

to_pil = transforms.ToPILImage()

for person in os.listdir(input_base):
    person_input_folder = os.path.join(input_base, person)
    person_output_folder = os.path.join(output_base, person)
    os.makedirs(person_output_folder, exist_ok=True)
    for img_name in os.listdir(person_input_folder):
        input_img_path = os.path.join(person_input_folder, img_name)
        output_img_path = os.path.join(person_output_folder, img_name)
        try:
            img = Image.open(input_img_path).convert('RGB')
            face = mtcnn(img)
            if face is not None:
                # Convert tensor (C, H, W) in [0,1] to PIL Image in RGB
                face = (face + 1) / 2
                face_img = to_pil(face.cpu().clamp(0, 1))
                # Convert to grayscale ('L' mode)
                # face_img_gray = face_img.convert('L')
                compress_image(face_img, output_img_path, max_size_kb=50)
            else:
                print(f"No face detected in {input_img_path}, skipping.")
        except Exception as e:
            print(f"Error processing {input_img_path}: {e}")
