# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from http import HTTPStatus

from starlette.responses import JSONResponse

from vllm.exceptions import GenerationError, RetryableRequestError
from vllm.logger import init_logger

from ..engine.protocol import ErrorInfo, ErrorResponse
from .utils import sanitize_message

logger = init_logger(__name__)


def create_error_response(
    message: str | Exception,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
    param: str | None = None,
) -> ErrorResponse:
    exc: Exception | None = None
    retry_after: int | None = None

    if isinstance(message, Exception):
        exc = message
        logger.debug(
            "create_error_response called with %s: %s", type(exc).__name__, exc
        )

        from vllm.exceptions import (
            GracefulHTTPError,
            VLLMClientError,
            VLLMNotFoundError,
            VLLMServerError,
            VLLMUnprocessableEntityError,
            VLLMValidationError,
        )

        if isinstance(exc, GracefulHTTPError):
            err_type = HTTPStatus(exc.http_status).phrase
            status_code = exc.http_status
            param = None
            retry_after = exc.retry_after
        elif isinstance(exc, VLLMValidationError):
            err_type = "BadRequestError"
            status_code = HTTPStatus.BAD_REQUEST
            param = exc.parameter
        elif isinstance(exc, VLLMUnprocessableEntityError):
            err_type = "UnprocessableEntityError"
            status_code = HTTPStatus.UNPROCESSABLE_ENTITY
            param = exc.parameter
        elif isinstance(exc, VLLMNotFoundError):
            err_type = "NotFoundError"
            status_code = HTTPStatus.NOT_FOUND
            param = None
        elif isinstance(exc, VLLMClientError):
            # Any other client-caused error defaults to 400.
            err_type = "BadRequestError"
            status_code = HTTPStatus.BAD_REQUEST
            param = None
        elif isinstance(exc, GenerationError):
            err_type = (
                "ServiceUnavailableError"
                if isinstance(exc, RetryableRequestError)
                else "InternalServerError"
            )
            status_code = exc.status_code
            param = None
        elif isinstance(exc, VLLMServerError):
            # Any other server-caused error defaults to 500.
            err_type = "InternalServerError"
            status_code = HTTPStatus.INTERNAL_SERVER_ERROR
            param = None
        # Fallback for raw exceptions not yet migrated to VLLMError.
        # TODO(zqzten): remove these fallback handlers after migration to VLLMError
        elif isinstance(exc, (ValueError, TypeError, OverflowError)):
            err_type = "BadRequestError"
            status_code = HTTPStatus.BAD_REQUEST
            param = None
        elif isinstance(exc, NotImplementedError):
            err_type = "NotImplementedError"
            status_code = HTTPStatus.NOT_IMPLEMENTED
            param = None
        elif any(cls.__name__ == "TemplateError" for cls in type(exc).__mro__):
            # jinja2.TemplateError and its subclasses (avoid importing jinja2)
            err_type = "BadRequestError"
            status_code = HTTPStatus.BAD_REQUEST
            param = None
        else:
            err_type = "InternalServerError"
            status_code = HTTPStatus.INTERNAL_SERVER_ERROR
            param = None

        message = str(exc)

    return ErrorResponse(
        error=ErrorInfo(
            message=sanitize_message(message),
            type=err_type,
            code=status_code.value,
            param=param,
            retry_after=retry_after,
        )
    )


def error_response_to_json_response(err: ErrorResponse) -> JSONResponse:
    """Render an ``ErrorResponse`` as a JSON HTTP response.

    Adds the ``Retry-After`` header when the error carries one, so that
    clients can back off without parsing the body.

    Args:
        err: The error payload to serialize.

    Returns:
        A JSON response with the error's HTTP status code.

    """
    headers = (
        {"Retry-After": str(err.error.retry_after)}
        if err.error.retry_after is not None
        else None
    )
    return JSONResponse(err.model_dump(), status_code=err.error.code, headers=headers)
