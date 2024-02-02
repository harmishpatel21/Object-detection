import streamlit as st
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw, ImageFont

processor = DetrImageProcessor.from_pretrained(f"./detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained(f"./detr-resnet-50", revision="no_timm")

st.set_page_config(layout = "wide")

def main():
    st.header("Object Detection")
    
    uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file)

        draw = ImageDraw.Draw(image)
        myFont = ImageFont.truetype("C:/Windows/Fonts/ariblk.ttf", 20)
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            print(
                    f"Detected {model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
            )
            st.write(
                    f"Detected {model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
            )
            draw.rectangle(box, outline = "red", width = 2)
            draw.text((box[0], box[1]), model.config.id2label[label.item()], font=myFont)
        with col2:
            st.image(image)

