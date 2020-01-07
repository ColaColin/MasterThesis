import importlib


def constructor_for_class_name(module_name):
    class_name = module_name.split(".")[-1]
    m = importlib.import_module(".".join(module_name.split(".")[:-1]))
    c = getattr(m, class_name)
    return c