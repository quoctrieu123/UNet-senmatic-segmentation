import argparse
import torch
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
import numpy as np

parser = argparse.ArgumentParser(description="Run inference on an image using a pretrained segmentation model.")
parser.add_argument('--image_path', type=str, required=True, help="Path to the input image.")
args = parser.parse_args()

model = smp.UnetPlusPlus(
    encoder_name="resnet34",      
    encoder_weights='imagenet',          
    in_channels=3,                 
    classes=3                     
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("Unetmodel.pth", map_location=device)  
if "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])  
else:
    model.load_state_dict(checkpoint)  

model.to(device)
model.eval()


preprocess = transforms.Compose([
    transforms.Resize((256, 256)),   
    transforms.ToTensor(),          
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

input_image = Image.open(args.image_path).convert("RGB")  
input_tensor = preprocess(input_image).unsqueeze(0).to(device)

with torch.inference_mode():
    output = model(input_tensor)
    output = torch.argmax(output, dim=1).squeeze().cpu().numpy()

colordict = {
    0: (0, 0, 0),       
    1: (255,0, 0),     
    2: (0,255, 0),     
}

def mask_to_rgb(mask, color_dict):
    output = np.zeros((mask.shape[0], mask.shape[1], 3))

    for k in color_dict.keys():
        output[mask==k] = color_dict[k]
    return np.uint8(output) 

segmented_color_image = mask_to_rgb(output,colordict)
segmented_color_image_pil = Image.fromarray(segmented_color_image)
segmented_color_image_pil.save("Output_Segmented_Image.png")
print("Segmented image with color saved as Output_Segmented_Image.png")