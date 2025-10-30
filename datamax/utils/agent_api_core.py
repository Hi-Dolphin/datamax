import pathlib
import sys
import httpx
import os
import functools
import json
import time
from httpx import URL

class APIModule:
    """
    中台数据服务模块
    """

    def __init__(self):
        self.API_CLIENT_ID = os.getenv("API_CLIENT_ID")
        self.API_CLIENT_SECRET = os.getenv("API_CLIENT_SECRET")
        self.API_HOST = os.getenv("API_HOST")
        self.API_BASE_PATH = os.getenv("API_BASE_PATH")
        self.API_OAUTH_TOKEN_URL: URL | str = os.getenv("API_OAUTH_TOKEN_URL")  # type: ignore
        self.access_token = None
        self.access_token_type = None
        self.access_token_expires_at: float | None = None

    def auth(self, force_refresh: bool = False) -> str:
        """
        中台数据服务授权
        """
        if (
            not force_refresh
            and self.access_token
            and self.access_token_type
            and (
                self.access_token_expires_at is None
                or self.access_token_expires_at > time.time()
            )
        ):
            return self.access_token

        auth_response = httpx.get(
            self.API_OAUTH_TOKEN_URL,
            params={
                "grant_type": "client_credentials",
                "client_id": self.API_CLIENT_ID,
                "client_secret": self.API_CLIENT_SECRET,
            },
            timeout=60,
        )

        auth_response.raise_for_status()
        try:
            response_body = auth_response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError("无法解析鉴权返回的JSON内容") from exc

        access_token = response_body.get("access_token")
        if not access_token:
            raise RuntimeError("鉴权返回缺少access_token字段")

        token_type = response_body.get("token_type") or "Bearer"
        expires_in = response_body.get("expires_in")

        expires_at: float | None = None
        if expires_in is not None:
            try:
                expires_seconds = float(expires_in)
            except (TypeError, ValueError):
                expires_seconds = None
            if expires_seconds is not None and expires_seconds > 0:
                expires_at = time.time() + max(0.0, expires_seconds - 5.0)

        self.access_token = access_token
        self.access_token_type = token_type
        self.access_token_expires_at = expires_at
        return self.access_token

    @staticmethod
    def requires_auth(func):
        """
        装饰器，用于检查是否已经授权，如果没有授权则调用授权方法
        """

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            token_initialized = bool(self.access_token and self.access_token_type)
            token_expired = (
                self.access_token_expires_at is not None
                and self.access_token_expires_at <= time.time()
            )
            if not token_initialized or token_expired:
                self.auth()
            return func(self, *args, **kwargs)

        return wrapper

    @requires_auth
    def call_api(
        self,
        method,
        endpoint,
        params=None,
        request_body=None,
    ) -> dict:
        """
        通用的API调用方法
        :method: 请求方法 get / post
        :endpoint: 请求路径
        :params: 请求参数 query
        :data: 请求数据 payload
        """

        url = f"{self.API_HOST}{self.API_BASE_PATH}{endpoint}"
        headers = {
            "Authorization": f"{self.access_token_type} {self.access_token}"
        }
        response = httpx.request(
            method,
            url,
            params=params,
            json=request_body,
            headers=headers,
            timeout=60,
        )
        # 不同data可能所属的类型不同，这里加上一一处理
        if (
            (isinstance(response.json()["data"], int))
            or (isinstance(response.json()["data"], str))
            or (isinstance(response.json()["data"], float))
        ):
            cleaned_dict = response.json()
        if isinstance(response.json()["data"], dict):
            cleaned_data = {}
            for k, v in response.json()["data"].items():
                if v is None:
                    continue
                cleaned_data[k] = v
            cleaned_dict = {"data": [cleaned_data]}
        # 过滤掉data中所有键值为空的键值对
        if (
            isinstance(response.json()["data"], list)
            and "data" in response.json()
        ):
            cleaned_data = [
                dict((k, v) for k, v in item.items() if v is not None)
                for item in response.json()["data"]
            ]
            cleaned_dict = {"data": cleaned_data}
        else:
            cleaned_dict = dict(response.json())
        return cleaned_dict

    @requires_auth
    def call_api_str(
        self,
        method,
        endpoint,
        params=None,
        request_body=None,
    ):
        """
        通用的API调用方法
        :method: 请求方法 get / post
        :endpoint: 请求路径
        :params: 请求参数 query
        :data: 请求数据 payload
        """

        url = f"{self.API_HOST}{self.API_BASE_PATH}{endpoint}"
        headers = {
            "Authorization": f"{self.access_token_type} {self.access_token}"
        }
        response = httpx.request(
            method,
            url,
            params=params,
            json=request_body,
            headers=headers,
            timeout=60,
        )
        # 不同data可能所属的类型不同，这里加上一一处理
        if response.json().get("data") == None:
            return json.dumps({"data": {}}, ensure_ascii=False)
        if (
            (isinstance(response.json()["data"], int))
            or (isinstance(response.json()["data"], str))
            or (isinstance(response.json()["data"], float))
        ):
            cleaned_dict = response.json()
        if isinstance(response.json()["data"], dict):
            cleaned_data = {}
            for k, v in response.json()["data"].items():
                if v is None:
                    continue
                cleaned_data[k] = v
            cleaned_dict = {"data": [cleaned_data]}
        # 过滤掉data中所有键值为空的键值对
        if (
            isinstance(response.json()["data"], list)
            and "data" in response.json()
        ):
            cleaned_data = [
                dict((k, v) for k, v in item.items() if v is not None)
                for item in response.json()["data"]
            ]
            cleaned_dict = {"data": cleaned_data}
        else:
            cleaned_dict = dict(response.json())
        return json.dumps(cleaned_dict, ensure_ascii=False)

    def __str__(self):
        """
        返回类的字符串表示
        """

        return f"<APIModule client_id={self.API_CLIENT_ID} client_secret={self.API_CLIENT_SECRET}>"

    def __repr__(self):
        """
        返回类的详细表示
        """
        return f"APIModule(API_CLIENT_ID={self.API_CLIENT_ID}, API_CLIENT_SECRET={self.API_CLIENT_SECRET})"

    def __enter__(self):
        """
        实现上下文管理器的进入方法
        """
        self.auth()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        实现上下文管理器的退出方法
        """
        # 这里可以释放资源或处理异常
        if exc_type:
            print(f"Exception type: {exc_type}")
            print(f"Exception value: {exc_val}")
            print(f"Exception traceback: {exc_tb}")
        self.access_token = None
        self.access_token_type = None


api_module = APIModule()
