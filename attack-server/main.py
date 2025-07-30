import io
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import torchvision
import base64
from celery_worker import sum_celery_task
from datetime import datetime

from utils import run_attack

class Item(BaseModel):
    attack_name: str
    image: str
    p: float = 2.0
    epsilon: float = 50.0
    max_iters: int = 30

class SumItem(BaseModel):
    x: int
    y: int

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "This is the nn_trust attack server"}

@app.post("/sum")
async def add(item: SumItem):
    d = datetime.now().isoformat()
    sum_celery_task.delay(item.x, item.y, d)
    return {"message": "Results will be written to file."}

@app.post("/attack")
async def attack(item: Item):

    img = Image.open(io.BytesIO(base64.b64decode(item.image)))
    img_tensor = torchvision.transforms.ToTensor()(img)
    x_adv, y, y_adv = run_attack(
        img=img_tensor,
        attack_name=item.attack_name,
        epsilon=item.epsilon,
        p=item.p,
        max_iters=item.max_iters
    )
    x_adv_img = torchvision.transforms.ToPILImage()(x_adv)

    buffered = io.BytesIO()
    x_adv_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())

    return {
        "attack_success": True,
        "x_adv": img_str,
        "y": y,
        "y_adv": y_adv
    }