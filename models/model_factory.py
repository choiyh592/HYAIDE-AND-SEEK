from .SANA import SANA
from .InstaFlow import InstaFlow
# TODO : Add imports

def create_model(args):
    model_name = args.model.lower()

    if model_name == "sana":
        return SANA()
    elif model_name == "instaflow":
        return InstaFlow()





