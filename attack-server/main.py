import io
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import torchvision
import base64

from utils import run_attack

class Item(BaseModel):
    attack_name: str
    image: str
    p: float = 2.0
    epsilon: float = 50.0

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "This is the nn_trust attack server"}


@app.post("/attack")
async def attack(item: Item):

    img = Image.open(io.BytesIO(base64.b64decode(item.image)))
    img_tensor = torchvision.transforms.ToTensor()(img)
    x_adv = run_attack(
        img=img_tensor,
        attack_name=item.attack_name,
        epsilon=item.epsilon,
        p=item.p
    )
    x_adv_img = torchvision.transforms.ToPILImage()(x_adv)

    buffered = io.BytesIO()
    x_adv_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())

    return {
        "attack_success": True,
        "x_adv": img_str
    }