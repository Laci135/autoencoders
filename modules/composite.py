import modules

class Composite(modules.Module):
    
    def __init__(self):
        __components = {}

    def add(self, name, component):
        self._components[name] = component

    def get(name):
        assert name in __components, f"Component not found: {name}"
        return __components[name]
