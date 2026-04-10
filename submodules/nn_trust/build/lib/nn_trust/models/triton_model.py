import numpy as np
import torch
from tritonclient.http import InferenceServerClient, InferInput

from nn_trust.core import Knowledge, ModelAdapter


class TritonClientModel(ModelAdapter):
    __torch_dtype_to_string_dict__ = {
        torch.uint8: "UINT8",
        torch.int8: "INT8",
        torch.int16: "INT16",
        torch.int32: "INT32",
        torch.int64: "INT64",
        torch.float16: "FP16",
        torch.float32: "FP32",
        torch.float64: "FP64",
    }

    __torch_dtype_to_numpy_dict__ = {
        torch.uint8: np.uint8,
        torch.int8: np.int8,
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
    }

    def __init__(
        self,
        server_addr: str,
        model_name: str,
        output_names: list[str] | None = None,
        input_names: list[str] | None = None,
    ):
        super().__init__(model=None, threat_model=Knowledge.Black)
        self.client = InferenceServerClient(server_addr)
        self.output_names = output_names
        self.input_names = input_names
        self.model_name = model_name

    def forward(
        self,
        *args: torch.Tensor,
    ) -> list[torch.Tensor] | torch.Tensor:
        inputs = []
        for name, arg in zip(self.input_names, args, strict=True): 
            inp = InferInput(name, arg.shape, TritonClientModel.__torch_dtype_to_string_dict__[arg.dtype])
            inp.set_data_from_numpy(
                arg.cpu().numpy().astype(TritonClientModel.__torch_dtype_to_numpy_dict__[arg.dtype])
            )
            inputs.append(inp)

        res = self.client.infer(self.model_name, inputs)
        res = [torch.from_numpy(res.as_numpy(out_name)) for out_name in self.output_names]
        # if the output is a single value, don't assume it belongs to a list
        if len(res) == 1:
            return res[0]
        return res
