import requests
import base64
from PIL import Image
import io
import json

imagefile = "../assets/data/ILSVRC2012_val_00000003_n02105855.JPEG"
with open(imagefile, "rb") as f:
    encoded_img = base64.b64encode(f.read()).decode("utf8")

img = Image.open(io.BytesIO(base64.b64decode(encoded_img)))

url = 'http://127.0.0.1:8000/attack'
attack_config = {
    'image': encoded_img, 
    "p":2.0, 
    "epsilon": 30.5, 
    "attack_name":"fgsm",
    "max_iters":10
    }
print(attack_config)
response = requests.post(url, json = attack_config)
response = json.loads(response.content)

img = Image.open(io.BytesIO(base64.b64decode(response["x_adv"])))
print(f'y: {response["y"]}, y_adv: {response["y_adv"]}')
img.show()
