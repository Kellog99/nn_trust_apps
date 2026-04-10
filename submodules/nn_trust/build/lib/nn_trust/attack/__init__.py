import importlib
import inspect
import pathlib
import pkgutil

from ._evasion import EvasionAttack, EvasionAttackConfig
from .attack_factory import EvasionAttackFactory

package_dir = pathlib.Path(__file__).resolve().parent / "evasion"  # Get evasion package directory
for _, module_name, _ in pkgutil.iter_modules([str(package_dir)]):  # List all modules and import
    module = importlib.import_module(f"{__name__}.evasion.{module_name}")
    for attribute_name in dir(module):  # List all attributes and only import the classes
        attribute = getattr(module, attribute_name)
        if inspect.isclass(attribute):
            globals()[attribute_name] = attribute
