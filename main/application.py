from common.controller import Controller
from common.model import Model
from common.view import View

def run():
    model = Model()
    controller = Controller(model)
    view = View(controller)
