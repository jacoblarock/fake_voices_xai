class container:
    """
    Class for easy wrapping and unwrapping of objects to avoid unexpected behavior
    """

    def __init__(self, data):
        self.data = data
        return

    def get_underlying(self):
        return self.data

def make_container(data) -> container:
    """
    Wrap prodived object in a container class
    """
    return container(data)

def get_underlying(con: container):
    """
    Function returns the underlying data in a provided container
    """
    return con.get_underlying()
