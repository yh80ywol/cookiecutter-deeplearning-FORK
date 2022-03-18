import abc, json

#Dataloader
class DataloaderBaseClass(abc.ABC):

    @abc.abstractmethod
    def __init__(self, config):
        self.config = config

    def execute_request(self, opt):
        raise NotImplementedError()   

    @abc.abstractmethod 
    def parse_data(self, data):
        raise NotImplementedError()

    @abc.abstractmethod
    def save_data(self, path, file_name, file_type, data):
        if file_type == "json":
            with open(path + file_name + ".json", "w") as f:
                json.dump(data, f, indent=2)
        elif file_type == "png":
            with open(path + file_name + ".png", "wb") as f:
                f.write(data)
        else:
            print("file_type not supported!")

    def __next__(self, start, end, data):
        current = start
        while current < len(data) and current < end:
            yield([data[current]])
            current += 1

    def __get__(self, data):
        self.parse_data(data)