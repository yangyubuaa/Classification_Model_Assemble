from utils.load import load_json

class TrainRecordVisualize:
    def __init__(self, train_record_json):

        self.train_params = load_json(train_record_json)


    def visualize(self):
        raise Exception("未完成！")