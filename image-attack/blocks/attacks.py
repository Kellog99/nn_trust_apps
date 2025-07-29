import gradio as gr
import torch
from pydantic_core import PydanticUndefined

from nn_trust.attack import EvasionAttackFactory
from nn_trust.core import ModelAdapter


class ListAttacks:
    def __init__(
        self,
        model: ModelAdapter,
        device: torch.device,
        task: int,
        labels_id: dict[int | str],
        max_number_parameters: int = 3,
        atk_not_valid: list[str] = ["hpuap", "parsimonious", "advyolo"],
    ):
        self.device = device
        self.max_number_parameters = max_number_parameters
        self.parameters = {"task": task, "model": model}
        # this dictionary save the position and the name of the configuration
        self.params_position = {i: None for i in range(max_number_parameters)}
        self.attack_dict = {
            repr(
                EvasionAttackFactory.create_attack(
                    atk_id, task=task, model=ModelAdapter(model=None, name="placeholder")
                )
            ): atk_id
            for atk_id in ["fgsm", "pgd", "deepfool"]
        }
        self.atk = ""
        self.labels_id = labels_id
        self.id_labels = {label: class_id for class_id, label in labels_id.items()}
        self.target_class = torch.tensor([0], device=self.device)

    def get_attack(self):
        """
        Generate the attack given the parameters.
        """
        atk_params = {}

        def annotated_value(value, annotation):
            return annotation(value) if annotation in [float, int, str, bool] else value

        for field_name, field_info in EvasionAttackFactory.list_config_param(self.atk).items():
            annotation = field_info.annotation
            # Convert the types
            if field_name in self.parameters:
                atk_params[field_name] = annotated_value(value=self.parameters[field_name], annotation=annotation)
            else:
                if field_info.default is not PydanticUndefined:
                    atk_params[field_name] = annotated_value(value=field_info.default, annotation=annotation)

        atk_params["device"] = torch.device("cuda" if torch.cuda.is_available() and self.parameters["gpu"] else "cpu")
        atk_params["verbose"] = True
        return EvasionAttackFactory.create_attack(self.atk, **atk_params)


    def get_config(self, attack_name: str):
        """
        Return the attack's parameters that can be modified in the UI
        """
        # saving the attack
        self.atk = self.attack_dict[attack_name]
        atk_params = EvasionAttackFactory.list_config_param(self.atk)
        atk_params = {k:atk_params[k] for k in ("max_iters", "p", "epsilon")}
        parameters = []
        i = 0
        for field_name, field_info in atk_params.items():
            if field_info.annotation in [float, int, str]:
                if len(parameters) >= self.max_number_parameters:
                    break
                params = {
                    "label": variable_to_human_readable_text(field_name),
                    "value": getattr(field_info, "default", None),
                    "visible": True,
                    "info": getattr(field_info, "description", None),
                }
                self.params_position[i] = field_name
                i = i + 1
                parameters.append(params)

        while len(parameters) < self.max_number_parameters:
            parameters.append({"visible": False})
            self.params_position[i] = None
            i = i + 1

        return parameters

    def update_config(self, atk: str):
        """
        Update the parameters in the UI with the new attack's parameters
        """
        parameters = self.get_config(atk)
        return [gr.update(**params) for params in parameters]

    def update_value(self, id, value):
        self.parameters.update({id: value})

    def add_slider(self, x):
        if "iter" in x.lower():
            return {
                "maximum": 80,
                "minimum": 1
            }
        elif "eps" in x.lower():
            return {
                "maximum": 100,
                "minimum": 0
            }
    def generate(self):
        default_config = list(self.attack_dict.keys())[0]
        with gr.Column():
            gr.Markdown("### 2. Select Attack 🖥️")
            attack = gr.Dropdown(
                choices=list(self.attack_dict),
                label="Select Algorithm",
                value=default_config,
                allow_custom_value=False,
                multiselect=False,
                interactive=True,
            )

            with gr.Accordion("Advance Settings", open=False):
                gr.Markdown("### 3. Attack Parameters ⚙️")
                # Create initialprint(param) parameters based on first attack
                params = []
                for param in self.get_config(default_config):
                    
                    if len(param) >=2:
                        slider = self.add_slider(param["label"])
                        if slider is not None:
                            params.append(gr.Slider(value=param["value"], info=param["info"], minimum=slider["minimum"], maximum=slider["maximum"], label=param["label"]))
                        else:
                            params.append(gr.Textbox(**param))

                targeted = gr.Checkbox(label="Targeted attack")
                choices = [*self.labels_id.values()]
                target = gr.Dropdown(
                    label="Select the Target class", choices=choices, value=choices[0], visible=False, interactive=True
                )

                gr.Markdown("### 4. System Parameters 💽")
                gpu = gr.Checkbox(label="Use Gpu.", value=True)
                self.update_value(id="gpu", value=True)

        ########## EVENT HANDLER ##########
        attack.change(fn=self.update_config, inputs=attack, outputs=params)

        def toogle_change(i: int, value):
            if self.params_position[i] is not None:
                self.update_value(self.params_position[i], value)

        for i, txt in enumerate(params):
            txt.change(fn=toogle_change, inputs=[gr.State(value=i), txt])

        def toogle_selection(id, value):
            self.update_value(id, value)
            return gr.update(visible=value)

        targeted.select(fn=toogle_selection, inputs=[gr.State(value="targeted"), targeted], outputs=target)

        gpu.select(fn=self.update_value, inputs=[gr.State(value="gpu"), gpu])

        target.change(
            fn=lambda x: setattr(self, "target_class", torch.tensor([self.id_labels[x]], device=self.device)),
            inputs=target,
        )

        return targeted, target


# Conversion of strings to be more readable
def variable_to_human_readable_text(variable_name: str) -> str:
    r"""
    Converts a snake_case variable name to a human-readable string.
    - Replaces underscores with spaces.
    - Capitalizes the first letter of each word.
    - Handles common abbreviations (can be extended).

    Args:
        variable_name (str): The snake_case variable name.

    Returns:
        str: The human-readable string.
    """
    # Handle common abbreviations (extend this dictionary as needed)
    common_abbreviations = {
        "num ": "number ",
        "id ": "ID ",
        "val ": "value ",
        "cnt ": "count ",
        "max ": "maximum ",
        "min ": "minimum ",
        "lr ": "learning rate ",
        "iters ": "iterations ",
        "avg ": "average ",
        "std ": "standard deviation",  # Example of multi-word  expansion
        "var ": "variance ",
        "cfg ": "configuration ",
        "ctx ": "context ",
        "idx ": "index ",
        "msg ": "message ",
        "rcv ": "received ",
        "snd ": "sent ",
        "tmp ": "temporary ",
        "ext ": "external ",
        "int ": "internal ",
        "ctrl ": "control ",
        "hdr ": "header ",
        "src ": "source ",
        "dst ": "destination ",
        "auth ": "authorization ",
        "param ": "parameter ",
        "ret ": "return ",
        "rsp ": "response ",
        "req ": "request ",
    }
    variable_name = variable_name.replace("_", " ")
    variable_name = variable_name + " "
    for abr in common_abbreviations:
        variable_name = variable_name.replace(abr, common_abbreviations[abr])
    variable_name = variable_name.title()
    return variable_name
