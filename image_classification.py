from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import time

# Start measuring the execution time
start_time = time.time()
path = r"C:\Users\Moksh\Dropbox\PC\Downloads\Men's-black-blank-T-shirt-template-on-transparent-background-PNG.png"
image = Image.open(path)

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

inputs = processor(images=image, return_tensors="pt")



outputs = model(**inputs)
logits = outputs.logits
print(logits)
predicted_class_idx = logits.argmax(-1).item()

# End measuring the execution time
end_time = time.time()
execution_time = end_time - start_time

print("Predicted class:", model.config.id2label[predicted_class_idx])
print("Execution time:", execution_time, "seconds")
