from ooad import Detect, Vehicle

class Detect_facade:
    def __init__(self, task):
        self.detect = Detect(task)

    def detect_engine(self):
        return self.detect.detect_engine()