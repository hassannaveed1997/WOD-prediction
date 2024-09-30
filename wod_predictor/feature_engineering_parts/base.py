class BaseFEPipelineObject:
    def __init__(self):
        self.meta_data = {}

    def transform(self, data):
        raise NotImplementedError
