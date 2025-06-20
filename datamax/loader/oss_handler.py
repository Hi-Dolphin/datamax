import datetime
import os
import subprocess

import oss2
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

load_dotenv()


def removing(path):
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            if dir == "__pycache__":
                pycache_path = os.path.join(root, dir)
                subprocess.run(["rm", "-rf", pycache_path], check=False)


def format_size_adaptive(value):
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = 1024.0
    for i in range(len(units)):
        if (value / size) < 1:
            return "%.2f%s" % (value, units[i])
        value = value / size


def format_datetime_into_isoformat(date_time: datetime.datetime) -> str:
    return (
        date_time.replace(tzinfo=datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


class OssClient:
    def __init__(
        self, oss_access_key_id, oss_access_key_secret, oss_endpoint, oss_bucket_name
    ):
        self.bucket_name = oss_bucket_name
        self.auth = oss2.Auth(
            os.getenv("OSS_ACCESS_KEY_ID", oss_access_key_id),
            os.getenv("OSS_ACCESS_KEY_SECRET", oss_access_key_secret),
        )
        self.endpoint = os.getenv("OSS_ENDPOINT", oss_endpoint)
        self.bucket = oss2.Bucket(
            self.auth, self.endpoint, os.getenv("OSS_BUCKET_NAME", oss_bucket_name)
        )

    # Upload a file
    # Usage: ossBucket.put_object_from_file("my-object-key", "path/to/local/file.txt")
    def put_object_from_file(self, object_name, file_path, progress_callback=None):
        self.bucket.put_object_from_file(
            object_name, file_path, progress_callback=progress_callback
        )

    # Download a file
    # Usage: ossBucket.get_object_to_file("my-object-key", "path/to/local/output-file.txt")
    def get_object_to_file(self, object_name, file_path, progress_callback=None):
        try:
            self.bucket.get_object_to_file(
                object_name, file_path, progress_callback=progress_callback
            )
        except oss2.exceptions.NoSuchKey:
            raise

            # Upload a folder

    # Usage: ossBucket.put_object_from_folder("my-object-folder", "path/to/local/folder")
    def put_pdf_word_from_folder(
        self, object_folder_name, local_folder_path, progress_callback=None
    ):
        for root, dirs, files in os.walk(local_folder_path):
            for file in tqdm(files, desc=root):
                if file.endswith(".pdf") or file.endswith(".word"):
                    file_path = os.path.join(root, file)
                    object_name = os.path.join(
                        object_folder_name, file_path[len(local_folder_path) + 1 :]
                    )
                    self.bucket.put_object_from_file(
                        object_name, file_path, progress_callback=progress_callback
                    )
                    # logger.info("object name: {}, file path: {}".format(
                    #     object_name, file_path))

    # Upload a folder
    # Usage: ossBucket.put_object_from_folder("my-object-folder", "path/to/local/folder")
    def put_object_from_folder(
        self, object_folder_name, local_folder_path, progress_callback=None
    ):
        for root, dirs, files in os.walk(local_folder_path):
            for file in tqdm(files, desc=root):
                file_path = os.path.join(root, file)
                object_name = os.path.join(
                    object_folder_name, file_path[len(local_folder_path) + 1 :]
                )
                self.bucket.put_object_from_file(
                    object_name, file_path, progress_callback=progress_callback
                )
                logger.info(
                    "object name: {}, file path: {}".format(object_name, file_path)
                )

    # Download a folder
    # Usage: ossBucket.get_object_to_folder("my-object-folder", "path/to/local/output-folder")
    def get_object_to_folder(
        self, object_folder_name, local_folder_path, progress_callback=None
    ):
        os.makedirs(local_folder_path, exist_ok=True)
        for obj in oss2.ObjectIterator(self.bucket, prefix=object_folder_name):
            file_path = os.path.join(
                local_folder_path, obj.key[len(object_folder_name) + 1 :]
            )
            self.bucket.get_object_to_file(
                obj.key, file_path, progress_callback=progress_callback
            )

    # Get all objects in the bucket
    # Usage: ossBucket.get_all_objects_in_bucket()
    def get_all_objects_in_bucket(self, prefix=None, delimiter=None):
        for obj in oss2.ObjectIterator(self.bucket, prefix=prefix, delimiter=delimiter):
            if obj.is_prefix():  # obj is folder
                logger.info("directory key: {}".format(obj.key))
            else:  # obj is file
                logger.info(
                    "file key: {}, object last modified: {}, object size: {}".format(
                        obj.key,
                        format_datetime_into_isoformat(
                            datetime.datetime.fromtimestamp(obj.last_modified)
                        ),
                        format_size_adaptive(obj.size),
                    )
                )

    def get_objects_in_folders(self, prefix: str):
        all_keys = []
        for obj in oss2.ObjectIterator(self.bucket, prefix=prefix):
            if obj.is_prefix():  # obj is folder
                pass
            else:  # obj is file
                if obj.key.endswith("/"):
                    continue
                all_keys.append(obj.key)
        return all_keys

    def delete_object(self, object_name="test"):
        if object_name is None or object_name == "":
            raise Exception(
                "Danger! object name is None or '' Will delete all objects in bucket!"
            )
        self.bucket.delete_object(key=object_name)

    # Delete a folder
    # Usage: ossBucket.delete_object_folder("my-object-folder")
    def delete_object_folder(self, object_folder_name="test"):
        if object_folder_name is None or object_folder_name == "":
            raise Exception(
                "Danger! object name is None or '' Will delete all objects in bucket!"
            )
        for obj in oss2.ObjectIterator(self.bucket, prefix=object_folder_name):
            self.bucket.delete_object(obj.key)
            logger.info("delete object key: {}".format(obj.key))

    def get_oss_url(
        self, object_name, url_expires_time, aliyun_oss_url_prefix, csnt_url_prefix
    ):
        oss_prefix = "oss://" + os.getenv("OSS_BUCKET_NAME", self.bucket_name) + "/"
        if object_name.__contains__(oss_prefix):
            object_name = object_name.replace(oss_prefix, "")
        aliyun_url = self.bucket.sign_url(
            "GET",
            object_name,
            int(os.getenv("URL_EXPIRES_TIME", url_expires_time)),
            slash_safe=True,
        )
        csnt_url = aliyun_url.replace(
            os.getenv("ALIYUN_OSS_URL_PREFIX", aliyun_oss_url_prefix),
            os.getenv("CSNT_URL_PREFIX", csnt_url_prefix),
        )
        return csnt_url

    def get_default_oss_url(self, object_name: str, url_expires_time):
        aliyun_url = self.bucket.sign_url(
            "GET",
            object_name,
            int(os.getenv("url_expires_time", url_expires_time)),
            slash_safe=True,
        )
        return aliyun_url
