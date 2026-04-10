from errno import EHOSTDOWN
from typing import Any, Tuple

import requests
import torch
import torch.nn as nn
from flask import Flask, request, jsonify

from nn_trust.core import AttackException, ModelAdapter, Task


class _APIModelAdapter(nn.Module):

    def __init__(self,
                 api_url: str,
                 device: torch.device = torch.device("cpu")
                 ) -> None:
        super().__init__()
        self.api_url = api_url
        self.device = device

    def check_health(self) -> bool:
        response = requests.get(f"{self.api_url}/health")
        return response.status_code == 200

    def forward(self, *inputs):
        # Convert input tensors to lists for JSON serialization
        input_data = {"inputs": [x.tolist() for x in inputs]}

        # Make API call for forward pass
        response = requests.post(f"{self.api_url}/forward", json=input_data)

        if response.status_code == 200:
            # Convert API response back to tensors
            output_data = response.json()['outputs']
            return tuple(torch.tensor(y, requires_grad=True, device=self.device) for y in output_data)
        else:
            raise AttackException(f"API forward call failed with status code {response.status_code}")

    def backward(self, grad_outputs, inputs):
        # Convert gradient tensors and inputs to lists for JSON serialization
        grad_data = {
            "grad_outputs": [g.tolist() for g in grad_outputs],
            "inputs": [x.tolist() for x in inputs]
        }

        # Make API call for backward pass
        response = requests.post(f"{self.api_url}/backward", json=grad_data)
        if response.status_code == 200:
            print(f"API Backward success! preparing output")
            # Convert API response back to tensors
            grad_data = response.json()['grad_inputs']
            print(f"API backward call succeeded with tensor on device {self.device}")
            return tuple(torch.tensor(g, device=self.device) for g in grad_data)
        else:
            raise AttackException(f"API backward call failed with status code {response.status_code}")

    def to(self, device):
        self.device = device


# Custom autograd function
class _APIModelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, adapter: _APIModelAdapter, *inputs: Any) -> Tuple[torch.Tensor, ...]:
        ctx.save_for_backward(*inputs)
        ctx.adapter = adapter
        return adapter.forward(*inputs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Tuple[None, ...]:
        inputs = ctx.saved_tensors
        grads = ctx.adapter.backward(grad_outputs, inputs)
        return (None,) + grads  # None for the adapter, then grads for each input


# Wrapper class
class APIModel(ModelAdapter):
    def __init__(self,
                 api_url: str,
                 device: torch.device = torch.device("cpu"),
                 task: str | Task = Task.from_str("classification")
                 ) -> None:
        """
        This Adapter use an api that calls post requests on '/forward' and '/backward' on a server.
        It hides the requests call from the rest of the code.

        :param api_url: The url where the model API is exposed.
        """
        super().__init__(model=None)
        self.api_url = api_url
        self.device = device
        self.adapter = _APIModelAdapter(api_url, device=self.device)
        print(f"Adapter url: {self.api_url} device: {self.device}")
        if not self.adapter.check_health:
            raise ValueError("Model failed to check api status")

    def forward(self, *inputs):
        return _APIModelFunction.apply(self.adapter, *inputs)

    def to(self, device: torch.device):
        self.adapter.to(device)


class FlaskAPIModelServer(object):
    """
    Flask Server that exposes the forward and backward method required from APIModel for an arbitrary pytorch model.
    """
    def __init__(self,
                 model: nn.Module,
                 host: str = "0.0.0.0",
                 port: int = 5000,
                 debug: bool = False,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 ) -> None:
        self._model = model
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        self.debug = debug
        self.device = device

        self.model = model.to(self.device)
        self.model.eval()

        # Setup routes
        @self.app.route('/', methods=['GET'])
        def home():
            return "FlaskAPIModelServer"

        @self.app.route('/health', methods=['GET'])
        def health():
            resp = jsonify(success=True)
            resp.status_code = 200
            return resp

        @self.app.route('/forward', methods=['POST'])
        def forward():
            return self.handle_forward()

        @self.app.route('/backward', methods=['POST'])
        def backward():
            return self.handle_backward()

    def handle_forward(self):
        data = request.json
        inputs = [torch.tensor(x, dtype=torch.float32, device=self.device) for x in data['inputs']]
        outputs = self._model(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        return jsonify({"outputs": [o.tolist() for o in outputs]})

    def handle_backward(self):
        data = request.json
        inputs = [torch.tensor(x, dtype=torch.float32, requires_grad=True, device=self.device) for x in data['inputs']]
        grad_outputs = [torch.tensor(g, dtype=torch.float32, device=self.device) for g in data['grad_outputs']]
        print({elem.device for elem in grad_outputs})
        # Perform forward pass
        outputs = self._model(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        print(f"Performing backward on the server")
        # Perform backward pass
        torch.autograd.backward(outputs, grad_outputs)
        print(f"Done backward on the server")
        print(f"This is grad sum: {[x.grad.sum() for x in inputs]}")
        return jsonify({
            "grad_inputs": [x.grad.tolist() for x in inputs]
        })

    def run(self):
        self.app.run(host=self.host, port=self.port, debug=self.debug)

    def get_url(self):
        return f"http://{self.host}:{self.port}"
