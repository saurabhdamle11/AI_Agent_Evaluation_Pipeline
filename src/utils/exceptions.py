class IngestionError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class KafkaPublishError(IngestionError):
    pass


class DatabaseError(IngestionError):
    pass
