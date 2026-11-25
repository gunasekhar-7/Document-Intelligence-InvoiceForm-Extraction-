"""MinIO/S3 client for document storage."""

import hashlib
import io
import mimetypes
from datetime import timedelta
from pathlib import Path
from typing import BinaryIO

from minio import Minio
from minio.error import S3Error

from document_intelligence.config import settings
from document_intelligence.logging import get_logger

logger = get_logger(__name__)


class MinioClient:
    """Client for MinIO object storage operations."""

    def __init__(
        self,
        endpoint: str = None,
        access_key: str = None,
        secret_key: str = None,
        secure: bool = None,
        bucket: str = None,
    ):
        self.endpoint = endpoint or settings.minio_endpoint
        self.access_key = access_key or settings.minio_root_user
        self.secret_key = secret_key or settings.minio_root_password
        self.secure = secure if secure is not None else settings.minio_secure
        self.bucket = bucket or settings.minio_bucket

        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure,
        )

        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        """Create bucket if it doesn't exist."""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info("Created bucket", bucket=self.bucket)
        except Exception as e:
            logger.warning(
                "Failed to ensure bucket or connect to MinIO", bucket=self.bucket, error=str(e)
            )

    def upload_file(
        self,
        object_key: str,
        data: BinaryIO,
        length: int,
        content_type: str = None,
        metadata: dict = None,
    ) -> str:
        """Upload file to MinIO."""
        if content_type is None:
            content_type = mimetypes.guess_type(object_key)[0] or "application/octet-stream"

        try:
            self.client.put_object(
                self.bucket,
                object_key,
                data,
                length,
                content_type=content_type,
                metadata=metadata or {},
            )
            logger.info("Uploaded file", bucket=self.bucket, key=object_key, size=length)
            return object_key
        except S3Error as e:
            logger.error("Failed to upload file", key=object_key, error=str(e))
            raise

    def upload_bytes(
        self,
        object_key: str,
        data: bytes,
        content_type: str = None,
        metadata: dict = None,
    ) -> str:
        """Upload bytes to MinIO."""
        return self.upload_file(
            object_key,
            io.BytesIO(data),
            len(data),
            content_type=content_type,
            metadata=metadata,
        )

    def download_file(self, object_key: str) -> bytes:
        """Download file from MinIO."""
        try:
            response = self.client.get_object(self.bucket, object_key)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as e:
            logger.error("Failed to download file", key=object_key, error=str(e))
            raise

    def delete_file(self, object_key: str) -> None:
        """Delete file from MinIO."""
        try:
            self.client.remove_object(self.bucket, object_key)
            logger.info("Deleted file", bucket=self.bucket, key=object_key)
        except S3Error as e:
            logger.error("Failed to delete file", key=object_key, error=str(e))
            raise

    def get_presigned_url(
        self,
        object_key: str,
        expires: timedelta = timedelta(hours=1),
        method: str = "GET",
    ) -> str:
        """Generate presigned URL for object access."""
        try:
            return self.client.presigned_get_object(
                self.bucket,
                object_key,
                expires=expires,
            )
        except S3Error as e:
            logger.error("Failed to generate presigned URL", key=object_key, error=str(e))
            raise

    def file_exists(self, object_key: str) -> bool:
        """Check if file exists in MinIO."""
        try:
            self.client.stat_object(self.bucket, object_key)
            return True
        except S3Error as e:
            if e.code == "NoSuchKey":
                return False
            raise

    def get_file_info(self, object_key: str) -> dict:
        """Get file metadata."""
        try:
            stat = self.client.stat_object(self.bucket, object_key)
            return {
                "size": stat.size,
                "last_modified": stat.last_modified,
                "etag": stat.etag,
                "content_type": stat.content_type,
                "metadata": stat.metadata,
            }
        except S3Error as e:
            logger.error("Failed to get file info", key=object_key, error=str(e))
            raise


def compute_content_hash(data: bytes) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(data).hexdigest()


def generate_object_key(filename: str, content_hash: str) -> str:
    """Generate unique object key for document."""
    ext = Path(filename).suffix
    return f"documents/{content_hash[:2]}/{content_hash}{ext}"


# Lazy global client
_minio_client_instance: MinioClient | None = None


def get_minio_client() -> MinioClient:
    """Get or create singleton MinioClient instance."""
    global _minio_client_instance
    if _minio_client_instance is None:
        _minio_client_instance = MinioClient()
    return _minio_client_instance


minio_client: MinioClient | None = None
