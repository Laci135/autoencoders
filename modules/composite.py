import modules

class Composite(modules.Module):
    
    def __init__(self):
        super(Composite, self).__init__()
        self.__components = {}

    def add(self, name, component):
        self.__components[name] = component

    def get(self, name):
        assert name in self.__components, f"Component not found: {name}"
        return self.__components[name]
