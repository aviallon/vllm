# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import Request

from vllm.logger import init_logger

from ..error_response import create_error_response, error_response_to_json_response

logger = init_logger(__name__)


async def exception_handler(req: Request, exc: Exception):
    if req.app.state.args.log_error_stack:
        logger.error(
            "Exception caught. Request id: %s",
            req.state.request_metadata.request_id
            if hasattr(req.state, "request_metadata")
            else None,
        )

    return error_response_to_json_response(create_error_response(exc))
