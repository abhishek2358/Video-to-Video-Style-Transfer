import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision import models
from PIL import Image

def decode_segmap(image, source, nc=21):
    label_colors = np.array(
        [
            (0, 0, 0),  
            (128, 0, 0),
            (0, 128, 0),
            (128, 128, 0),
            (0, 0, 128),
            (128, 0, 128),
            (0, 128, 128),
            (128, 128, 128),
            (64, 0, 0),
            (192, 0, 0),
            (64, 128, 0),
            (192, 128, 0),
            (64, 0, 128),
            (192, 0, 128),
            (64, 128, 128),
            (192, 128, 128),
            (0, 64, 0),
            (128, 64, 0),
            (0, 192, 0),
            (128, 192, 0),
            (0, 64, 128),
        ]
    )
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    rgb = np.stack([r, g, b], axis=2)

    foreground = cv2.imread(source)
    foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    foreground = cv2.resize(foreground,(r.shape[1],r.shape[0]))
    
    background = 255 * np.ones_like(rgb).astype(np.uint8)
    
    foreground = foreground.astype(float)
    background = background.astype(float)
    
    th, alpha = cv2.threshold(np.array(rgb),0,255, cv2.THRESH_BINARY)
    
    
    #alpha = cv2.GaussianBlur(alpha, (7,7),0) # If want to apply a slight blur to the mask to soften edges
    
    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float)/255
    
    foreground = cv2.multiply(alpha, foreground)
    
    background = cv2.multiply(1.0 - alpha, background)
    
    outImage = cv2.add(foreground, background)
    
    return outImage/255

def segment(net, path, source):
    img = Image.open(path)
    trf = T.Compose([T.Resize(256), 
                    T.CenterCrop(224), 
                    T.ToTensor(), 
                    T.Normalize(mean = [0.485, 0.456, 0.406], 
                                std = [0.229, 0.224, 0.225])])
    trf_v2 = T.Compose([T.Resize(256), 
                    T.CenterCrop(224), 
                    T.ToTensor()])
    inp = trf(img).unsqueeze(0)
    out = net(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    rgb = decode_segmap(om, source)
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    foreground_mask=np.where(rgb==1,1,0)
    background_mask=np.where(rgb==1,0,1)
    inp_img=trf_v2(img).permute(1, 2, 0)    
    foreground_img=foreground_mask*inp_img.numpy()
    background_img=background_mask*inp_img.numpy()
    plt.imshow(foreground_img)
    plt.subplot(1, 3, 3)
    plt.imshow(background_img)
    plt.axis('off')
    plt.show()
    input()
    return foreground_img,background_img,foreground_mask,background_mask

import tensorflow as tf
import tensorflow_hub as hub
def apply_style(f_img,b_img,style_img,f_mask,b_mask):
   
    f_img_blur = cv2.GaussianBlur(f_img, (35,35),0) ##Blur out foreground
    b_img_blur = cv2.GaussianBlur(b_img, (35,35),0) ##Blur out background
    f_img=f_img+b_mask*b_img_blur
    b_img=b_img+f_mask*f_img_blur

    plt.subplot(1, 2, 1)
    plt.imshow(f_img)
    plt.subplot(1, 2, 2)
    plt.imshow(b_img)
    plt.axis('off')
    plt.show()
    
    content_image = f_img
    style_image=style_img
    style_image=cv2.GaussianBlur(style_image, (35,35),0)
    
    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
   
    style_image = tf.image.resize(style_image, (224, 224))

    hub_module = hub.load('https://kaggle.com/models/google/arbitrary-image-stylization-v1/frameworks/TensorFlow1/variations/256/versions/1')

    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    f_stylized_image = outputs[0].numpy().squeeze(0)
    content_image = b_img
    style_image=style_img
    style_image=cv2.GaussianBlur(style_image, (35,35),0)

    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
    
    style_image = tf.image.resize(style_image, (224, 224))

    hub_module = hub.load('https://kaggle.com/models/google/arbitrary-image-stylization-v1/frameworks/TensorFlow1/variations/256/versions/1')

    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    b_stylized_image = outputs[0].numpy().squeeze(0)

    output_image=b_stylized_image*b_mask + f_stylized_image*f_mask
    print(type(b_stylized_image))

    print(f_stylized_image)
    plt.subplot(1, 3, 1)
    plt.imshow(f_stylized_image)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(b_stylized_image)
    plt.subplot(1, 3, 3)
    plt.imshow(output_image)
    plt.show()

    

    

dlab = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

folder_path = r"C:\AbhiData\Desktop\UPenncourses\CIS 581\CV Final Project\Fore-Background\Input Frame"
img_name_list = os.listdir(folder_path)

img_path = [os.path.join(folder_path, name) for name in img_name_list]
style_image_path=r"C:\AbhiData\Desktop\UPenncourses\CIS 581\CV Final Project\Neural Style Transfer\style-transfer-video-processor\style_ref\00.jpg"
style_image = plt.imread(style_image_path)
# Load the image
for i in range(len(img_path)):
    img = Image.open(img_path[i])
    print ('Segmenatation Image on DeepLabv3')
    f_img,b_img,f_mask,b_mask=segment(dlab, path=img_path[i], source=img_path[i])
    apply_style(f_img,b_img,style_image,f_mask,b_mask)
    




