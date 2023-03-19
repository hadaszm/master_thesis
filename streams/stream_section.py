class StreamSection:

    """
    A class for storing the stream with information if it is fully labelled
    """

    def __init__(self,name, stream, is_fully_labelled) -> None:
        self.stream = stream
        self.is_fully_supervised = is_fully_labelled
        self.__name__ = name
