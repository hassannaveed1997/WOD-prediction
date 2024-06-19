class BaseFEPipelineObject:
    def __init__(self):
        pass

    def transform(self, data):
        raise NotImplementedError
