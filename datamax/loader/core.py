import os
from typing import List

from datamax.loader.minio_handler import MinIOClient
from datamax.loader.oss_handler import OssClient


class DataLoader:
    def __init__(
        self,
        endpoint: str = None,
        secret_key: str = None,
        access_key: str = None,
        bucket_name: str = None,
        source: str = None,
    ):
        if source and source == "Oss":
            self.oss = OssClient(
                oss_endpoint=endpoint,
                oss_access_key_secret=secret_key,
                oss_access_key_id=access_key,
                oss_bucket_name=bucket_name,
            )
        elif source and source == "MinIO":
            self.mi = MinIOClient(
                endpoint=endpoint,
                secret_key=secret_key,
                access_key=access_key,
                bucket_name=bucket_name,
            )
        self.download_path = str("./download_file")
        self.source = source
        self.bucket_name = bucket_name

    @staticmethod
    def load_from_file(local_file_path) -> List[str]:
        if os.path.isfile(local_file_path):
            if os.path.exists(local_file_path):
                if os.access(local_file_path, os.R_OK):
                    return [local_file_path]
                else:
                    return []
            else:
                return []
        elif os.path.isdir(local_file_path):
            access_path = []
            # Recursively process all files and subdirectories under the current directory.
            for item in os.listdir(local_file_path):
                item_path = os.path.join(local_file_path, item)
                item_results = DataLoader.load_from_file(item_path)
                access_path.extend(item_results)
            return access_path
        else:
            return []

    def load_from_oss_source(self, oss_path: str) -> List[str]:
        if not os.path.exists(self.download_path):
            os.makedirs(self.download_path)

        self.download(oss_path=oss_path)

        file_list = []
        for root, dirs, files in os.walk(self.download_path):
            for file in files:
                file_path = os.path.join(self.download_path, file)
                file_list.append(file_path)

        success_file_list = []
        for file_path in file_list:
            if self.load_from_file(file_path):
                success_file_list.append(file_path)

        return success_file_list

    def download(self, oss_path: str):
        if self.source == "MinIO":
            file_list = self.mi.list_objects(
                bucket_name=self.bucket_name, prefix=oss_path
            )
            for path in file_list:
                self.mi.download_file(
                    bucket_name=self.bucket_name,
                    object_name=path,
                    file_path=f'{self.download_path}/{path.split("/")[-1]}',
                )
        elif self.source == "Oss":
            keys = self.oss.get_objects_in_folders(prefix=oss_path)
            for path in keys:
                self.oss.get_object_to_file(
                    object_name=path,
                    file_path=f'{self.download_path}/{path.split("/")[-1]}',
                )

    def upload(self, local_file_path: str, save_prefix: str):
        if self.source == "MinIO":
            if os.path.isdir(local_file_path):
                for root, dirs, files in os.walk(local_file_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        self.mi.upload_file(
                            bucket_name=self.bucket_name,
                            object_name=save_prefix + f"{file}",
                            file_path=file_path,
                        )
            elif os.path.isfile(local_file_path):
                self.mi.upload_file(
                    bucket_name=self.bucket_name,
                    object_name=save_prefix + os.path.basename(local_file_path),
                    file_path=local_file_path,
                )
            else:
                pass

        elif self.source == "Oss":
            if os.path.isdir(local_file_path):
                self.oss.put_object_from_folder(
                    object_folder_name=save_prefix, local_folder_path=local_file_path
                )
            elif os.path.isfile(local_file_path):
                self.oss.put_object_from_file(
                    object_name=save_prefix + os.path.basename(local_file_path),
                    file_path=local_file_path,
                )
        else:
            pass

    def share(
        self,
        oss_path: str,
        expires: int = None,
        aliyun_oss_url_prefix: str = None,
        csnt_url_prefix: str = None,
    ):
        if self.source == "MinIO":
            return self.mi.get_object_tmp_link(
                bucket_name=self.bucket_name, object_name=oss_path, expires=expires
            )
        elif self.source == "Oss":
            return self.oss.get_oss_url(
                object_name=oss_path,
                url_expires_time=expires,
                aliyun_oss_url_prefix=aliyun_oss_url_prefix,
                csnt_url_prefix=csnt_url_prefix,
            )
