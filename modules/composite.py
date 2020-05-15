import modules

# abstract class for composite modules (may contain other modules)
class Composite(modules.Module):
    
    def __init__(self):
        super(Composite, self).__init__()
        self.__components = {} # dict of components
    
    # add a named component
    def add(self, name, component):
        self.__components[name] = component

    # get a component by name
    def get(self, name):
        assert name in self.__components, f"Component not found: {name}"
        return self.__components[name]
