class BaseProvider:
    def chat(self, message, **kwargs):
        raise NotImplementedError("The chat method must be implemented by subclasses.")

    @property
    def name(self):
        raise NotImplementedError("The name property must be implemented by subclasses.")