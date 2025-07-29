import requests
import base64
from PIL import Image
import io

imagefile = "./assets/data/sample_images/f35_1.jpg"

with open(imagefile, "rb") as f:
    encoded_img = str(base64.b64encode(f.read()))

img = Image.open(io.BytesIO(base64.b64decode(encoded_img)))

url = 'http://127.0.0.1:8000/attack'
attack_config = {'image': encoded_img, "p":2.0, "epsilon": 30.5, "attack_name":"fgsm"}

x = requests.post(url, json = attack_config)

print(x)