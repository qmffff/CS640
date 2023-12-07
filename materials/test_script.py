import os, sys, pandas, pathlib, time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2
from torchvision.io import read_image

model_path = "absolute_path/to/your/model" # change this
all_labels = np.load("all_labels.npy").tolist()
test_image_dir = "test_images_compressed_80"
df_test = pandas.read_csv("test.csv")
model = torch.load(model_path)

# change the size only if necessary
transforms_test = v2.Compose([v2.Resize(256, antialias = True),
                              v2.CenterCrop(224),
                              v2.ToImage(),
                              v2.ToDtype(torch.float32, scale = True),
                              v2.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

YPredict = []
for id in tqdm(df_test["image_id"]):
    image = read_image(os.path.join(test_image_dir, str(id) + ".jpg"))
    with torch.no_grad():
        transformed_image = transforms_test(image).unsqueeze(dim = 0)
        # change the following line depending on your model's forward function
        _, logits = model(transformed_image)
        YPredict.append(np.argmax(logits.detach().cpu().numpy(), axis = 1).item())

YTrue = [all_labels.index(label) for label in df_test["label"]]
print("Confusion matrix: " + str(confusion_matrix(YTrue, YPredict)))
print("Accuracy: " + str(accuracy_score(YTrue, YPredict)))
print("F1: " + str(f1_score(YTrue, YPredict, average = "macro")))