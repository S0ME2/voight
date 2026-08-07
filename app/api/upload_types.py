"""Reusable HTTP upload parameter types.

FastAPI 0.129.1+ emits OpenAPI 3.1 ``contentMediaType`` for ``UploadFile``.
Swagger UI 5.x currently does not render array items declared that way as file
pickers. ``WithJsonSchema`` keeps runtime validation as ``UploadFile`` while
forcing the legacy ``format: binary`` schema that Swagger UI understands.

Keep this workaround isolated here so it can be removed when Swagger UI adds
full support for binary array items declared with ``contentMediaType``.
"""

from typing import Annotated, TypeAlias

from fastapi import File, UploadFile
from pydantic import WithJsonSchema

BATCH_FILE_DESCRIPTION = (
    "Select one or more image files, ZIP archives, or a mixture of both. "
    "ZIP archives may contain images in nested folders."
)

SwaggerUploadFile: TypeAlias = Annotated[
    UploadFile,
    WithJsonSchema(
        {
            "type": "string",
            "format": "binary",
            "contentMediaType": "application/octet-stream",
        }
    ),
]

SingleUpload: TypeAlias = Annotated[SwaggerUploadFile, File()]
OptionalUpload: TypeAlias = Annotated[SwaggerUploadFile | None, File()]
BatchUploads: TypeAlias = Annotated[
    list[SwaggerUploadFile],
    File(description=BATCH_FILE_DESCRIPTION),
]
