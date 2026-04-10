import pathlib

import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch

import numpy as np

from nn_trust.core import ModelAdapter, Knowledge, Task



class ONNXModel(ModelAdapter):
    r"""A general Open Neural Network Exchange (ONNX) models adapter.

    .. Note:: The input of a model should be batched, although a batched input is not supported in general because of
        the ONNX format restrictions.

    :param model_filepath: the filepath ot an onnx model.
    :param expected_output: either a :class:`torch.Tensor` or a :class:`list[torch.Tensor]`. It is required to be of the
        smae shape of the model's output.
    :param device: The device to use for the inference with the ONNX model.
    :param np_dtype: The `numpy` dtype to use for the conversion and internal representation of the computations.

    Example::

    In this example, we load a model used for CIFAR10 classification.

    >>> model = ONNXModel('PATH_TO_ONNX_MODEL.onnx', expected_output=torch.empty(1, 10), device='cpu')
    >>> model.forward(torch.rand(1, 3, 32, 32))
    {'last_layer_name': tensor([[-3.3755, -1.7209,  4.2052,  0.5517,  1.6866, -1.0264,  6.0363, -2.5375,
          -2.6358, -1.1840]])}
    """

    __numpy_to_torch_dtype_dict__ = {
            np.uint8: torch.uint8,
            np.int8: torch.int8,
            np.int16: torch.int16,
            np.int32: torch.int32,
            np.int64: torch.int64,
            np.float16: torch.float16,
            np.float32: torch.float32,
            np.float64: torch.float64,
            np.complex64: torch.complex64,
            np.complex128: torch.complex128
        }

    torch_to_numpy_precision = {torch.float32: np.float32, torch.float64: np.float64, torch.float16: np.float16}

    def __init__(
        self,
        model_filepath: str | pathlib.Path,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        np_dtype: type = np.float32,
        task: Task | str = Task.from_str("classification")
    ):

        super().__init__(model=None, threat_model=Knowledge.Black)
        self._device = device
        if np_dtype not in self.__numpy_to_torch_dtype_dict__:
            raise ValueError(f"The given np_dtype '{np_dtype}' is not supported. ")
        self._dtype = np_dtype
        self.torch_dtype = self.__numpy_to_torch_dtype_dict__[self._dtype]

        # Checks if the model is available/exists or is well-formed.
        self._model = onnx.load(model_filepath)
        if onnx.checker.check_model(self._model):
            raise ValueError(f"The ONNX model '{model_filepath}' was not loaded correctly.")

        # Create an ONNX runtime session
        self.providers = ONNXModel._from_device_to_execution_provider(device)
        self._session = ort.InferenceSession(path_or_bytes=model_filepath, providers=[self.providers])

        # create output shape
        self._expected_out = [
            {
                "name": out.name,
                "shape": out.shape,
                "type": out.type,
                "tensor": torch.empty(out.shape, dtype=self.torch_dtype, device=torch.device(self._device))
            }
            for out in self._session.get_outputs()
        ]

        self._expected_inputs = [
            {
                "name":out.name,
                "shape": out.shape,
                "type": out.type
            }
            for out in self._session.get_inputs()
        ]

    @staticmethod
    def _from_device_to_execution_provider(device: str = "cpu"):
        """Converts a torch device name string to an ExecutionProvider available in ONNX Runtime."""
        _inference_provider = device.upper() + "ExecutionProvider"
        if _inference_provider not in ort.get_available_providers():
            raise ValueError(
                f"The given provider, '{_inference_provider}', is not available."
                f"Available providers are: {ort.get_available_providers()}"
            )
        return _inference_provider

    def run_with_io_binding(
        self,
        *args: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        io_binding = self._session.io_binding()

        if len(args) != len(self._expected_inputs):
            raise ValueError(
                f"The number of arguments does not coincides with the expected number of model's inputs. Expected {self._expected_inputs}.")
        input_list = [{"tensor":tensor, **input_info} for tensor,input_info in zip(args,self._expected_inputs)]

        for expected_input in input_list:
            io_binding.bind_input(
                name=expected_input["name"],
                device_type=self._device,
                device_id=0,  # TODO: this could lead to issues in case of more than 1 device available!
                element_type=self._dtype,
                shape=tuple(expected_input["tensor"].shape),
                buffer_ptr=expected_input["tensor"].to(self.torch_dtype).data_ptr(),
            )

        for expected_out in self._expected_out:
            io_binding.bind_output(
                name=expected_out["name"],
                device_type=self._device,
                device_id=0,  # TODO: this could lead to issues in case of more than 1 device available!
                element_type=self._dtype,
                shape=expected_out["shape"],
                buffer_ptr=expected_out["tensor"].to(self.torch_dtype).data_ptr(),
            )

        self._session.run_with_iobinding(io_binding)
        res = io_binding.get_outputs()

        output_dict = {
            k.name: torch.from_numpy(v.numpy()).to(self._device).to(self.torch_dtype)
            for k, v in zip(self._session.get_outputs(), res)
        }
        
        return output_dict["logits"]

    def run_without_io_binding(
        self,
        *args: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if len(args) != len(self._expected_inputs):
            raise ValueError(
                f"Number of arguments does not match expected inputs: {len(self._expected_inputs)}"
            )

        # Prepare input dict for ONNX Runtime
        input_dict = {}
        for tensor, input_info in zip(args, self._expected_inputs):
            # ORT expects numpy arrays
            x_np = tensor.to("cpu").numpy()  # move to CPU
            input_dict[input_info["name"]] = x_np

        # Run inference
        output_np_list = self._session.run(None, input_dict)

        # Convert outputs back to PyTorch tensors, move to the desired device
        outputs = {}
        for output_info, out_np in zip(self._session.get_outputs(), output_np_list):
            out_tensor = torch.from_numpy(out_np).to(self._device).to(self.torch_dtype)
            outputs[output_info.name] = out_tensor

        return outputs

    def forward(
        self,
        *args: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.run_with_io_binding(*args)