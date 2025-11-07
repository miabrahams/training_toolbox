

class ExtractionFailedError(Exception):
    """Raised when extraction of fields from a prompt fails."""
    def __init__(self, message="Extraction of fields from prompt failed"):
        self.message = message
        super().__init__(message)

