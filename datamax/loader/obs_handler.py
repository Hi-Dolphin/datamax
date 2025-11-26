import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

try:
    from obs import ObsClient as HuaweiObsAPI  # type: ignore[import]
except ImportError as exc:  # pragma: no cover - communicated during initialization
    HuaweiObsAPI = None  # type: ignore[assignment]
    OBS_IMPORT_ERROR = exc
else:
    OBS_IMPORT_ERROR = None


class ObsClient:
    def __init__(
        self,
        obs_endpoint: str | None,
        obs_access_key_id: str | None,
        obs_access_key_secret: str | None,
        obs_bucket_name: str | None,
    ) -> None:
        if HuaweiObsAPI is None:
            raise ImportError(
                "Huawei OBS SDK is not installed. "
                "Install the 'esdk-obs-python' package to enable OBS support."
            ) from OBS_IMPORT_ERROR

        self.endpoint = os.getenv("OBS_ENDPOINT", obs_endpoint)
        self.access_key = os.getenv("OBS_ACCESS_KEY_ID", obs_access_key_id)
        self.secret_key = os.getenv("OBS_ACCESS_KEY_SECRET", obs_access_key_secret)
        self.bucket_name = os.getenv("OBS_BUCKET_NAME", obs_bucket_name)
        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        if HuaweiObsAPI is None:
            raise ImportError(
                "Huawei OBS SDK is not installed. "
                "Install the 'esdk-obs-python' package to enable OBS support."
            ) from OBS_IMPORT_ERROR

        return HuaweiObsAPI(
            access_key_id=self.access_key,
            secret_access_key=self.secret_key,
            server=self.endpoint,
        )

    def list_objects(self, prefix: str | None = None) -> list[str]:
        response = self.client.listObjects(
            bucketName=self.bucket_name, prefix=prefix or ""
        )
        if response.status >= 300:
            message = getattr(response, "errorMessage", "Failed to list OBS objects.")
            raise RuntimeError(message)

        contents = getattr(response.body, "contents", []) if response.body else []
        return [
            content.key
            for content in contents
            if getattr(content, "key", "").strip() and not content.key.endswith("/")
        ]

    def download_file(self, object_name: str, file_path: str) -> str:
        response = self.client.getObject(
            bucketName=self.bucket_name,
            objectKey=object_name,
            downloadPath=file_path,
        )
        if response.status >= 300:
            message = getattr(
                response,
                "errorMessage",
                f"Failed to download OBS object {object_name}.",
            )
            raise RuntimeError(message)
        return file_path

    def upload_file(self, file_path: str, object_name: str) -> None:
        response = self.client.putFile(
            bucketName=self.bucket_name,
            objectKey=object_name,
            file_path=file_path,
        )
        if response.status >= 300:
            message = getattr(
                response, "errorMessage", f"Failed to upload OBS object {object_name}."
            )
            raise RuntimeError(message)

    def get_object_tmp_link(self, object_name: str, expires: int | None = None) -> str:
        expiry = expires if expires and expires > 0 else 3600
        response = self.client.createSignedUrl(
            method="GET",
            bucketName=self.bucket_name,
            objectKey=object_name,
            expires=expiry,
        )
        signed_url = getattr(response, "signedUrl", None)
        if response.status >= 300 or not signed_url:
            message = getattr(
                response,
                "errorMessage",
                f"Failed to create OBS signed URL for {object_name}.",
            )
            raise RuntimeError(message)
        return signed_url
