from algorithm.mp_fedbase import MPBasicServer, MPBasicServer


class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
    
    def run(self):
        super().run()
        return


class Client(MPBasicServer):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)

