import pkgutil
import importlib
import sys

# Import all modules from the current package
__all__ = []
package_name = __name__

__version__ = "0.1.0"


for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
    # Dynamically import the module
    module = importlib.import_module(f"{package_name}.{module_name}")
    # Add the module to the global namespace
    globals()[module_name] = module
    __all__.append(module_name)
