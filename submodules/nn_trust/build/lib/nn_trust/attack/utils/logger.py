import abc
import logging
import queue
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Any, Optional

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """A default logger that does not log data. It provides a simple interface composed by three methods:
    - :meth:`log`: logs the information provided with a given name.
    - :meth:`step`: increases the internal state step counters by ``1``.
    - :meth:`close`: closes the logging object. For example, if a deamon thread is spawned when the logger is created
        this method force the closure of the thread and force correct saving of the data.

    Any :class:`Logger` has a dictionary of internal counter states that may be updated only via
    :meth:`step` or :meth:`step_all`.

    .. Note: In case the internal counter for a given state needs to be fixed, don't pass the state of the respective counter.

    .. Example

    Creates a logger that logs by increasing the internal step counter for ExampleLogger.

    >>> ex_logger = ExampleLogger(["training"])
    >>> # Logs the data at training/SOME_IMG_DATA at step = 0
    >>> ex_logger.log(tag="SOME_IMG_DATA", data=torch.rand(3, 128, 128), state="training", metadata="image")
    >>> # Do other stuff...
    >>> # Increase the counter by running step
    >>> ex_logger.step("training")
    >>> # Logs the data at training/SOME_IMG_DATA at step = 1
    >>> ex_logger.log(tag="SOME_IMG_DATA", data=torch.rand(3, 128, 128), state="training", metadata="image")

    """

    def __init__(self, states: Optional[list[str]] = None):
        r"""Initialize an instance of :class:`Logger`.

        :param states: counter names for the :class:`Logger`'s internal states.
        """
        if states is None:
            self._step_state = {}
        else:
            self._step_state = {s: 0 for s in states}

        if type(self) is Logger:
            logging.debug("You are using the default Logger class which does not log.")

    def set_step(self, state: str, step: int):
        if state in self._step_state:
            self._step_state[state] = step
        else:
            logging.debug(f"The logger state [{state}] is not defined.")

    def log(
            self,
            tag: str,
            data: Any,
            state: Optional[str] = None,
            metadata: Optional[str] = None
    ):
        """Logs the given data with additional metadata.

        :param tag: Name of the variable to log.
        :param data: Value to log.
        :param state: Name of the internal :class:`Logger`'s step counter. Default is ``None``.
        :param metadata: Additional information for the logger.

        :returns: None.
        """
        if state is None or state not in self._step_state:
            logging.debug(
                f"The logger state [{state}] does not exist in the current logger. Continue logging without increasing the step counter.")

    def step(self, state: Optional[str] = None):
        """Increases the internal step counter with the given name `state`.

        .. Note: If the `state` is ``None``, then all internal counters are updated.

        :param state: Name of the step counter. Default is ``None``.
        """
        if state is None:
            self.step_all()
        else:
            if state in self._step_state:
                self._step_state[state] += 1
            else:
                logging.debug(f"The state [{state}] is not defined.")

    def step_all(self):
        for k in self._step_state:
            self._step_state[k] += 1

    @abc.abstractmethod
    def close(self):
        """Eventually it closes the logger."""


class TensorboardLogger(Logger):
    def __init__(
            self,
            path: Path,
            states: Optional[list[str]] = None,
            image_transform: Optional[torchvision.transforms.transforms] = None,
    ):
        super().__init__(states)
        path.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._writer = SummaryWriter(path)
        self._transform = image_transform

    def log(self, tag: str, data: Any, state: Optional[str] = None, metadata: Optional[str] = None):
        super().log(tag, data, state, metadata)
        tag = tag if state is None else f"{state}/" + tag
        step = 0 if state is None or state not in self._step_state else self._step_state[state]
        if metadata is not None and metadata.startswith("image"):
            if self._transform is not None:
                data = self._transform(data)

        data_val = (tag, data, step)
        if metadata is None:
            if isinstance(data, torch.Tensor):
                log_method = self._writer.add_tensor
            elif isinstance(data, float | int):
                log_method = self._writer.add_scalar
            elif isinstance(data, str):
                log_method = self._writer.add_text
            else:
                raise ValueError(f"The type {type(data)} is not supported, provide 'metadata'.")
        else:
            if metadata in ["image", "images", "figure", "tensor", "scalar", "scalars", "text", "histogram"]:
                if metadata == "scalar" and isinstance(data_val[1], torch.Tensor):
                    # Additional conversion in case of a simple scalar
                    data_val = (data_val[0], data_val[1].item(), data_val[2])
                log_method = getattr(self._writer, f"add_{metadata}")
            else:
                raise ValueError(f"The given 'metadata'={metadata} is not available to {self.__class__.__name__}.")

        log_method(*data_val)

    def close(self):
        self._writer.flush()
        self._writer.close()


class PyTorchCheckpointLogger(Logger):
    r"""Stores the logged data using :func:`torch.save` in a daemon :class:`Thread` using a :class:`Queue`."""

    def __init__(
            self,
            path: Path,
            max_size: int = 10,
            save_interval: float = 3.0,
            flush_timeout: float = 3.0,
            states: Optional[list[str]] = None,
    ):
        r"""Initialize a :class:`PyTorchCheckpointLogger`.

        :param path:
        :param max_size:
        :param save_interval:
        :param flush_timeout:
        :param states:
        """
        super().__init__(states)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._path.touch(exist_ok=True)
        self._max_size = max_size
        self._save_interval = save_interval
        self._flush_timeout = flush_timeout
        self._queue = Queue(maxsize=max_size)

        self._running = True

        # Start the background saving thread
        self._save_thread = threading.Thread(target=self._save_loop)
        self._thread_lock = threading.Lock()
        self._save_thread.daemon = False
        self._save_thread.start()

    def _save_loop(self):
        """Background thread that periodically saves the buffer."""
        while self._running:
            time.sleep(self._save_interval)
            if not self._queue.empty():
                self._force_save()

    def _force_save(self):
        """Force save all buffered data."""
        self._thread_lock.acquire()
        if self._path.stat().st_size > 0:
            data = torch.load(str(self._path), weights_only=False)
        else:
            data = {}

        while not self._queue.empty():
            element_to_add = self._queue.get()
            tag, data_elem, step = element_to_add
            has_key_data = data.get(tag, None) is not None
            if has_key_data:
                # append the data depending on whether the step size is larger than the length
                if step >= len(data.get(tag, [])):
                    data[tag].append(data_elem)
                # insert the element at the given step
                else:
                    data[tag][step] = data_elem
            # Create a new tag at the given index
            else:
                data[tag] = [data_elem]

        if data:
            torch.save(data, self._path)
        self._thread_lock.release()

    def log(self, tag: str, data: Any, state: Optional[str] = None, metadata: Optional[str] = None):
        super().log(tag, data, state, metadata)
        # Copy the data to cpu to not cause big oofs on the cuda device
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu()

        tag = tag if state is None else f"{state}/{tag}"
        step = 0 if state is None or state not in self._step_state else self._step_state[state]
        data_val = (tag, data, step)
        try:
            self._queue.put(data_val, block=False)
        except queue.Full:
            # Handle queue full case
            self._force_save()
            self._queue.put(data_val, block=False)
        except queue.ShutDown:
            # Handle queue Shutdown case
            self.close()

    def close(self):
        self._running = False
        self._force_save()
        self._save_thread.join(timeout=self._flush_timeout)
