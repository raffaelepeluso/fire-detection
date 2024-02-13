import cv2
import os
import argparse
import random
import numpy as np
import torch
from torchvision import transforms
import torchvision.models as models
import albumentations as A


def init_parameter():
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument("--videos", type=str, default="foo_videos/", help="Dataset folder")
    parser.add_argument("--results", type=str, default="foo_results/", help="Results folder")
    args = parser.parse_args()
    return args

args = init_parameter()

# Here you should initialize your method

def preprocess_image(img):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB conversion
    img_array = np.array(rgb_img, dtype=np.uint8)  # NumPy conversion

    # trasformations according to the training model
    preprocessing = A.Compose(
        [
            A.Resize(height=224, width=224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
        ]
    )

    preprocessed_img = preprocessing(image=img_array)["image"]  # apply transformations
    tensor_img = transforms.ToTensor()(preprocessed_img)  # tensor conversion

    return tensor_img

# load the pre-trained model
state_dict_path = "model.pth"
model = models.mobilenet_v3_large(pretrained=True)
model.classifier[3] = torch.nn.Linear(in_features=1280, out_features=1, bias=True)
state_dict = torch.load(state_dict_path)
model.load_state_dict(state_dict)
model.eval()

threshold = 3 # for the consecutive fire frames

################################################

# For all the test videos
for video in os.listdir(args.videos):
    # Process the video
    ret = True
    cap = cv2.VideoCapture(os.path.join(args.videos, video))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    i = 0  # frame counter
    pos_neg = 0
    consecutive_fire = 0
    first_frame = 0

    while ret:
        ret, img = cap.read()
        # Here you should add your code for applying your method
        if not ret:
            break
        i += 1

        if i % frame_rate != 0 and i != 1:  # take also the first frame
            continue

        input_tensor = preprocess_image(img)
        input_batch = input_tensor.unsqueeze(0)  # batch for the model

        if (torch.cuda.is_available()):  # move the input and model to GPU for speed if available
            input_batch = input_batch.to("cuda")
            model.to("cuda")

        with torch.no_grad():
            output = model(input_batch)

        prob = torch.sigmoid(output)  # apply activation
        digit = (prob >= 0.5).int()

        if digit == 1:
            if consecutive_fire == 0:  # save the first frame in the consecutive window
                first_frame = i
            consecutive_fire += 1
            if consecutive_fire >= threshold:
                pos_neg = 1
                break
        else:
            consecutive_fire = 0

        ########################################################
    cap.release()
    f = open(args.results + video + ".txt", "w")
    # Here you should add your code for writing the results
    if pos_neg:
        t = int(first_frame / frame_rate) # output in seconds
        f.write(str(t))
    ########################################################
    f.close()
