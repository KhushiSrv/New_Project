import requests
from PIL import Image
from transformers import AutoProcessor,BlipForConditionalGeneration
import gradio as gr
import numpy as np 

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")              #generatring the caption
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")   #encoding and decoding

def image_cap(input_image:np.ndarray):
    raw_image = Image.fromarray(input_image).convert("RGB")
    inputs=processor(raw_image, return_tensors='pt')
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens = True)
    return caption

iface = gr.Interface(fn = image_cap, inputs = gr.Image(), output = "text", title = "Image Captioning Application", 
                     description = "Simple Web Application to generate Caption for the images.")

iface.launch(server_name="127.0.0.1", server_port=7860)