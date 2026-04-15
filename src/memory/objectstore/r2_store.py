from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import ObjectStoreConfig
from .provider import ObjectStoreProvider

# Import logger - assumes this module is imported as part of the package
try:
    from src.logging.logger import AgentLogger
except ImportError:
    import logging
    AgentLogger = logging.getLogger


class R2ObjectStore(ObjectStoreProvider):
    """Cloudflare R2 object store implementation using boto3 S3-compatible API."""

    def __init__(self, config: ObjectStoreConfig) -> None:
        self._config = config
        self._logger = AgentLogger()

        # Get R2 configuration from environment variables
        self._account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        self._access_key_id = os.getenv("R2_ACCESS_KEY_ID")
        self._secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")
        self._bucket_name = os.getenv("R2_BUCKET_NAME", config.r2_bucket_name)

        # Validate required configuration
        if not all([self._account_id, self._access_key_id, self._secret_access_key, self._bucket_name]):
            raise ValueError(
                "R2 configuration missing. Required: CLOUDFLARE_ACCOUNT_ID, R2_ACCESS_KEY_ID, "
                "R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME"
            )

        # Construct R2 endpoint URL
        self._endpoint_url = f"https://{self._account_id}.r2.cloudflarestorage.com"

        # Configure boto3 client with retries
        self._client = boto3.client(
            "s3",
            endpoint_url=self._endpoint_url,
            aws_access_key_id=self._access_key_id,
            aws_secret_access_key=self._secret_access_key,
            region_name="auto",  # R2 uses "auto" for region
            config=Config(
                retries={
                    "max_attempts": 3,
                    "mode": "standard"
                }
            )
        )

        self._logger.log(f"R2ObjectStore initialized for bucket: {self._bucket_name}", level="info")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ClientError, ConnectionError)),
        reraise=True
    )
    def write(self, key: str, data: bytes) -> str:
        """
        Upload data to R2 bucket.

        Args:
            key: Object key in format "{document_id}/{image_id}.{extension}"
            data: Binary data to upload

        Returns:
            Public URL of the uploaded object

        Raises:
            ClientError: If upload fails after retries
        """
        try:
            self._client.put_object(
                Bucket=self._bucket_name,
                Key=key,
                Body=data
            )

            # Construct public URL
            public_url = f"https://pub-{self._bucket_name}.r2.dev/{key}"

            self._logger.log(f"Successfully uploaded object to R2: {key} -> {public_url}", level="info")
            return public_url

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            self._logger.log(f"R2 upload failed for key '{key}': {error_code} - {str(e)}", level="error")
            raise
        except NoCredentialsError:
            self._logger.log(f"R2 upload failed: missing credentials for key '{key}'", level="error")
            raise

    def generate_presigned_url(self, key: str, expiration: int = 3600) -> str:
        """
        Generate a presigned URL to share an R2 object.

        Args:
            key: Object key.
            expiration: Time in seconds for the presigned URL to remain valid. Default is 1 hour.

        Returns:
            Presigned URL as a string.
        """
        try:
            url = self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket_name, "Key": key},
                ExpiresIn=expiration,
            )
            self._logger.log(f"Generated presigned URL for {key}", level="info")
            return url
        except ClientError as e:
            self._logger.log(f"Failed to generate presigned URL for {key}: {e}", level="error")
            raise
        except NoCredentialsError:
            self._logger.log(f"Presigned URL generation failed: missing credentials for key '{key}'", level="error")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ClientError, ConnectionError)),
        reraise=True
    )
    def read(self, key: str) -> bytes:
        """
        Download data from R2 bucket.

        Args:
            key: Object key in format "{document_id}/{image_id}.{extension}" or full R2 URL

        Returns:
            Binary data from the object

        Raises:
            FileNotFoundError: If object does not exist
            ClientError: If download fails after retries
        """
        # Extract key from URL if a full URL is provided
        object_key = key
        if key.startswith("http"):
            parsed = urlparse(key)
            # R2 public URL format: https://pub-{bucket}.r2.dev/{key}
            # We need to handle both public and presigned URLs potentially
            if "r2.cloudflarestorage.com" in parsed.netloc: # For non-public R2 URLs
                object_key = "/".join(parsed.path.split("/")[2:]) # This will depend on the exact URL structure
            else: # For pub-bucket.r2.dev URLs
                object_key = parsed.path.lstrip("/")

        try:
            response = self._client.get_object(
                Bucket=self._bucket_name,
                Key=object_key
            )
            data = response["Body"].read()

            self._logger.log(f"Successfully read object from R2: {object_key}", level="info")
            return data

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code in ["404", "NoSuchKey"]:
                self._logger.log(f"R2 object not found: {object_key}", level="warning")
                raise FileNotFoundError(f"Object not found in R2: {object_key}")
            self._logger.log(f"R2 download failed for key '{object_key}': {error_code} - {str(e)}", level="error")
            raise
        except NoCredentialsError:
            self._logger.log(f"R2 download failed: missing credentials for key '{object_key}'", level="error")
            raise
