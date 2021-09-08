class MetricsManager:

    def __init__(self, experiment):

        self.experiment = experiment

        self.data = dict()

    def append(self, new_data: list()):
        # environment and Buffer type as identifier
        self.data["/".join(new_data[:2])] = new_data[2:]

    def get_data(self, env, buffer_type, userun):
        userun = str(userun)
        return self.data["/".join([env, buffer_type, userun])]

    def recode(self, userun):
        for key in list(self.data.keys()):
            try:
                self.data["/".join([key, f"{userun}"])] = self.data.pop(key)
            except KeyError:
                pass
