"""Error types, each mapped to a message that is safe to show the candidate."""


class AppError(Exception):
    """Base class for all errors the app expects and handles."""

    user_message = "Something went wrong. Please try again."


class ExternalAPIError(AppError):
    """An external service (OpenRouter, embeddings API) failed or timed out."""

    user_message = "An external service is not responding right now. Please try again shortly."


class DataValidationError(AppError):
    """Stored or generated data does not have the expected shape."""

    user_message = "Some data could not be validated. Details were written to the log."


class UserInputError(AppError):
    """The candidate's message or arguments cannot be processed as given."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.user_message = message


class ConfigError(AppError):
    """The app is misconfigured (missing key, missing file); it cannot start."""

    user_message = "The app is not configured correctly. See the README section 'Configure'."
