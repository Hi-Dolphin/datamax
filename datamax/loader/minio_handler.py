import os
import re
from datetime import timedelta

from dotenv import load_dotenv
from loguru import logger
from minio import Minio
from minio.commonconfig import Tags
from minio.error import S3Error

load_dotenv()


class MinIOClient:
    def __init__(self, endpoint, access_key, secret_key, bucket_name, secure=False):
        self.endpoint = os.getenv("ENDPOINT_MINIO", endpoint)
        self.access_key = os.getenv("ACCESS_KEY_MINIO", access_key)
        self.secret_key = os.getenv("SECRET_KEY_MINIO", secret_key)
        self.bucket_name = os.getenv("BUCKET_NAME_MINIO", bucket_name)
        self.secure = secure
        self.client = self._initialize_client()

    def _initialize_client(self):
        try:
            client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure,
            )
            return client
        except S3Error as e:
            raise

    @staticmethod
    def bytes_to_mb(bytes_value):
        return bytes_value / (1024 * 1024)

    def create_bucket(self, bucket_name):
        if self.client:
            try:
                self.client.make_bucket(bucket_name)
                logger.info(f"Bucket '{bucket_name}' created successfully.")
            except S3Error as e:
                raise

    def remove_bucket(self, bucket_name):
        if self.client:
            try:
                self.client.remove_bucket(bucket_name)
                logger.info(f"Bucket '{bucket_name}' removed successfully.")
            except S3Error as e:
                raise

    def upload_file(self, file_path, bucket_name, object_name):
        if self.client:
            try:
                self.client.fput_object(bucket_name, object_name, file_path)
                logger.info(
                    f"File '{file_path}' uploaded to bucket '{bucket_name}' as '{object_name}'."
                )
            except S3Error as e:
                raise

    def download_file(self, bucket_name, object_name, file_path):
        if self.client:
            try:
                self.client.fget_object(bucket_name, object_name, file_path)
                logger.info(
                    f"Object '{object_name}' from bucket '{bucket_name}' downloaded to '{file_path}'."
                )
                return file_path
            except Exception as e:
                try:
                    illegal_chars = r'[\/:*?"<>|]'
                    file_path = re.sub(illegal_chars, "_", file_path)
                    self.client.fget_object(bucket_name, object_name, file_path)
                    logger.info(
                        f"Object {object_name} from bucket {bucket_name} downloaded to {file_path}'."
                    )
                    return file_path
                except Exception as e:
                    raise

    def list_objects(self, bucket_name, prefix=None):
        if self.client:
            try:
                result_list = []
                if prefix:
                    objects = self.client.list_objects(
                        bucket_name, recursive=True, prefix=prefix
                    )
                else:
                    objects = self.client.list_objects(bucket_name, recursive=True)
                logger.info(f"Objects in bucket '{bucket_name}':")
                for obj in objects:
                    result_list.append(obj.object_name)
                return result_list
            except S3Error as e:
                raise

    def remove_object(self, bucket_name, object_name):
        if self.client:
            try:
                self.client.remove_object(bucket_name, object_name)
            except S3Error as e:
                raise

    def calculate_bucket_stats(self, bucket_name, prefix):
        objects = self.client.list_objects(bucket_name, prefix=prefix, recursive=True)
        total_size = 0
        object_count = 0

        for obj in objects:
            object_count += 1
            total_size += obj.size

        total_size = self.bytes_to_mb(total_size)

        return object_count, total_size

    def get_objects(self, bucket_name, object_name):
        try:
            response = self.client.get_object(bucket_name, object_name)
            content = response.read().decode("utf-8")
            return content
        except Exception as e:
            raise

    def get_object_tag(self, bucket_name, object_name):
        try:
            tags = self.client.get_object_tags(
                bucket_name=bucket_name, object_name=object_name
            )
            return tags
        except Exception as e:
            raise

    def update_object_tag(self, bucket_name, object_name, tags):
        try:
            tags_obj = Tags.new_object_tags()
            tag_info = self.get_object_tag(
                bucket_name=bucket_name, object_name=object_name
            )
            if tag_info is None:
                tag_info = {}
                for tag_dict in tags:
                    for tag_key, tag_value in tag_dict.items():
                        if tag_key in tag_info:
                            tag_info[tag_key] = tag_value
                        else:
                            tag_info[tag_key] = tag_value

                for k, v in tag_info.items():
                    tags_obj[k] = v
                self.client.set_object_tags(
                    bucket_name=bucket_name, object_name=object_name, tags=tags_obj
                )
            else:
                for tag_dict in tags:
                    for tag_key, tag_value in tag_dict.items():
                        if tag_key in tag_info:
                            tag_info[tag_key] = tag_value
                        else:
                            tag_info[tag_key] = tag_value

                for k, v in tag_info.items():
                    tags_obj[k] = v
                self.client.set_object_tags(
                    bucket_name=bucket_name, object_name=object_name, tags=tags_obj
                )
            return tag_info
        except Exception as e:
            raise

    def reset_object_tag(self, bucket_name, object_name):
        try:
            self.client.delete_object_tags(
                bucket_name=bucket_name, object_name=object_name
            )
            return True
        except Exception as e:
            raise

    def get_object_tmp_link(self, bucket_name, object_name, expires):
        try:
            return self.client.presigned_get_object(
                bucket_name, object_name, expires=timedelta(days=expires)
            )
        except Exception as e:
            raise
