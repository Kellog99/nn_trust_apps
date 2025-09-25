from typing import List
from pydantic import BaseModel
from models import AttackProps, ParametersProps

def get_attacks() -> List[AttackProps]:
    """Return a hardcoded list of AttackProps (from the uploaded message.json)."""
    return [
        AttackProps(
            id="fgsm",
            knowledge="white",
            name="Fast Gradient Sign Method (FGSM)",
            description=(
                "A single-step adversarial attack that uses the gradient of the loss "
                "function to generate adversarial examples. It adds noise in the "
                "direction of the gradient to fool the model."
            ),
            parameters=[
                ParametersProps(
                    name="epsilon",
                    label="Epsilon",
                    min=0.001,
                    max=0.5,
                    step=0.001,
                    default=0.03,
                    description="Maximum perturbation magnitude allowed for each pixel",
                ),
                ParametersProps(
                    name="norm",
                    label="Norm Type",
                    min=1.0,
                    max=2.0,
                    step=1.0,
                    default=2.0,
                    description="Type of norm used (L1 or L2)",
                ),
            ],
        ),
        AttackProps(
            id="pgd",
            knowledge="black",
            name="Projected Gradient Descent (PGD)",
            description=(
                "An iterative adversarial attack that applies multiple small perturbations "
                "projected onto an epsilon ball. More powerful than FGSM as it uses multiple iterations."
            ),
            parameters=[
                ParametersProps(
                    name="epsilon",
                    label="Epsilon",
                    min=0.001,
                    max=0.5,
                    step=0.001,
                    default=0.03,
                    description="Maximum perturbation magnitude allowed",
                ),
                ParametersProps(
                    name="alpha",
                    label="Step Size (Alpha)",
                    min=0.0001,
                    max=0.1,
                    step=0.0001,
                    default=0.01,
                    description="Step size for each iteration",
                ),
                ParametersProps(
                    name="steps",
                    label="Number of Steps",
                    min=1.0,
                    max=100.0,
                    step=1.0,
                    default=40.0,
                    description="Number of gradient descent steps",
                ),
                ParametersProps(
                    name="random_start",
                    label="Random Start",
                    min=0.0,
                    max=1.0,
                    step=1.0,
                    default=1.0,
                    description="Whether to start from a random point (0 = false, 1 = true)",
                ),
            ],
        ),
        AttackProps(
            id="cw",
            knowledge="white",
            name="Carlini & Wagner (C&W)",
            description=(
                "An optimization-based attack that finds minimal perturbations by solving "
                "a constrained optimization problem. Known for generating high-quality adversarial examples."
            ),
            parameters=[
                ParametersProps(
                    name="c",
                    label="Confidence Parameter",
                    min=0.1,
                    max=100.0,
                    step=0.1,
                    default=1.0,
                    description="Confidence parameter that controls the trade-off between perturbation size and attack success",
                ),
                ParametersProps(
                    name="kappa",
                    label="Kappa",
                    min=0.0,
                    max=100.0,
                    step=1.0,
                    default=0.0,
                    description="Minimum confidence gap for successful attack",
                ),
                ParametersProps(
                    name="steps",
                    label="Optimization Steps",
                    min=100.0,
                    max=10000.0,
                    step=100.0,
                    default=1000.0,
                    description="Number of optimization steps",
                ),
                ParametersProps(
                    name="lr",
                    label="Learning Rate",
                    min=0.001,
                    max=1.0,
                    step=0.001,
                    default=0.01,
                    description="Learning rate for the optimization process",
                ),
            ],
        ),
        AttackProps(
            id="deepfool",
            knowledge="white",
            name="DeepFool",
            description=(
                "An iterative attack that finds the minimal perturbation needed to cross "
                "the decision boundary by approximating the classifier with a linear model at each step."
            ),
            parameters=[
                ParametersProps(
                    name="steps",
                    label="Maximum Steps",
                    min=1.0,
                    max=1000.0,
                    step=1.0,
                    default=50.0,
                    description="Maximum number of iterations",
                ),
                ParametersProps(
                    name="overshoot",
                    label="Overshoot",
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    default=0.02,
                    description="Overshoot parameter to ensure crossing the boundary",
                ),
            ],
        ),
        AttackProps(
            id="boundary",
            knowledge="black",
            name="Boundary Attack",
            description=(
                "A black-box attack that starts from an adversarial example and walks along "
                "the decision boundary to find closer adversarial examples to the original input."
            ),
            parameters=[
                ParametersProps(
                    name="steps",
                    label="Attack Steps",
                    min=100.0,
                    max=50000.0,
                    step=100.0,
                    default=25000.0,
                    description="Number of attack iterations",
                ),
                ParametersProps(
                    name="spherical_step",
                    label="Spherical Step Size",
                    min=0.001,
                    max=0.1,
                    step=0.001,
                    default=0.01,
                    description="Step size for spherical steps",
                ),
                ParametersProps(
                    name="source_step",
                    label="Source Step Size",
                    min=0.001,
                    max=0.1,
                    step=0.001,
                    default=0.01,
                    description="Step size for source steps",
                ),
            ],
        ),
    ]

