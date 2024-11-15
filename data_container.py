class container:

    def __init__(self, data):
        self.data = data
        return

    def get_underlying(self):
        return self.data

def make_container(data) -> container:
    return container(data)

def get_underlying(con: container):
    return con.get_underlying()
