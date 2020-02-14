import falcon

class LoginResource():
    def __init__(self, password):
        self.password = password

    def on_post(self, req, resp):
        if (req.media == self.password):
            resp.status = falcon.HTTP_200
        else:
            resp.status = falcon.HTTP_401