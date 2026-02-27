from .SANA import SANA
from .InstaFlow import InstaFlow
# TODO : Add imports

class FlowModels:
    def __init__(self, args):
        model_name = args.model_id.lower()

        models = {
            "sana": SANA,
            "instaflow": InstaFlow,
        }

        if model_name not in models:
            raise ValueError(f"Model {args.model_id} not supported. Choose from {list(models.keys())}")
        
        self._model = models[model_name]()
    
    @property
    def model(self):
        return self._model



        




