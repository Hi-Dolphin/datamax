import os
import requests
import mimetypes
# import magic  # 需要安装python-magic
from urllib.parse import urlparse, parse_qs, unquote
import tempfile
import shutil
import base64
import zipfile  # 用于二次验证 Office 文档
import filetype
import re
from loguru import logger


def _handle_special_urls(url):
    """
    处理特殊类型的URL，返回规范化的URL或特殊处理标志
    
    Args:
        url (str): 原始URL
        
    Returns:
        dict: 包含处理结果的字典
    """
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    
    # 处理常见的云存储和CDN服务
    special_handling = {
        'needs_auth': False,
        'url': url,
        'headers': {},
        'method_preference': ['head', 'range', 'stream'],
        'storage_type': 'generic',
        'requires_special_download': False
    }
    
    # 阿里云OSS检测和处理
    if 'aliyuncs.com' in domain or 'oss-' in domain:
        logger.info(f"🔍 检测到阿里云OSS URL")
        special_handling['storage_type'] = 'aliyun_oss'
        special_handling['requires_special_download'] = True
        
        # OSS特殊请求头
        special_handling['headers'].update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        })
        
        # OSS通常支持HEAD请求
        special_handling['method_preference'] = ['head', 'range', 'stream']
        
        # 检查是否为内网域名
        if '-internal.' in domain:
            logger.warning(f"⚠️ 检测到OSS内网域名，可能需要VPC环境访问")
    
    # 华为云OBS检测和处理
    elif 'obs.' in domain and 'myhuaweicloud.com' in domain:
        logger.info(f"🔍 检测到华为云OBS URL")
        special_handling['storage_type'] = 'huawei_obs'
        special_handling['requires_special_download'] = True
        
        special_handling['headers'].update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        })
        
        # OBS通常支持HEAD请求
        special_handling['method_preference'] = ['head', 'range', 'stream']
    
    # 腾讯云COS检测和处理
    elif 'cos.' in domain and 'myqcloud.com' in domain:
        logger.info(f"🔍 检测到腾讯云COS URL")
        special_handling['storage_type'] = 'tencent_cos'
        special_handling['requires_special_download'] = True
        
        special_handling['headers'].update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        })
        
        special_handling['method_preference'] = ['head', 'range', 'stream']
    
    # AWS S3检测和处理
    elif ('s3.amazonaws.com' in domain or 
          's3-' in domain or 
          '.s3.' in domain or 
          domain.endswith('.amazonaws.com')):
        logger.info(f"🔍 检测到AWS S3 URL")
        special_handling['storage_type'] = 'aws_s3'
        special_handling['requires_special_download'] = True
        
        special_handling['headers'].update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
        })
        
        special_handling['method_preference'] = ['head', 'range', 'stream']
    
    # Google Cloud Storage检测和处理
    elif 'storage.googleapis.com' in domain or 'storage.cloud.google.com' in domain:
        logger.info(f"🔍 检测到Google Cloud Storage URL")
        special_handling['storage_type'] = 'google_gcs'
        special_handling['requires_special_download'] = True
        
        special_handling['headers'].update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
        })
        
        special_handling['method_preference'] = ['head', 'range', 'stream']
    
    # 七牛云检测和处理
    elif any(provider in domain for provider in ['qiniu.com', 'qiniucdn.com', 'clouddn.com']):
        logger.info(f"🔍 检测到七牛云存储URL")
        special_handling['storage_type'] = 'qiniu'
        special_handling['requires_special_download'] = True
        
        special_handling['headers'].update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
        })
        
        special_handling['method_preference'] = ['head', 'range', 'stream']
    
    # 又拍云检测和处理
    elif any(provider in domain for provider in ['upyun.com', 'upaiyun.com']):
        logger.info(f"🔍 检测到又拍云存储URL")
        special_handling['storage_type'] = 'upyun'
        special_handling['requires_special_download'] = True
        
        special_handling['headers'].update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
        })
        
        special_handling['method_preference'] = ['head', 'stream']  # 又拍云可能不支持Range
    
    # Google Drive 文件
    elif 'drive.google.com' in domain:
        logger.info(f"🔍 检测到Google Drive URL")
        if '/file/d/' in url:
            file_id = url.split('/file/d/')[1].split('/')[0]
            # 转换为直接下载链接
            special_handling['url'] = f"https://drive.google.com/uc?export=download&id={file_id}"
            special_handling['headers']['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            special_handling['storage_type'] = 'google_drive'
    
    # OneDrive 文件
    elif 'onedrive.live.com' in domain or '1drv.ms' in domain:
        logger.info(f"🔍 检测到OneDrive URL")
        special_handling['headers']['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        special_handling['method_preference'] = ['stream']  # OneDrive 通常不支持HEAD
        special_handling['storage_type'] = 'onedrive'
    
    # Dropbox 文件
    elif 'dropbox.com' in domain:
        logger.info(f"🔍 检测到Dropbox URL")
        if 'dl=0' in url:
            special_handling['url'] = url.replace('dl=0', 'dl=1')
        special_handling['headers']['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        special_handling['storage_type'] = 'dropbox'
    
    # 百度网盘等需要特殊处理的服务
    elif any(service in domain for service in ['pan.baidu.com', 'lanzous.com', 'lanzoux.com']):
        logger.warning(f"🔒 检测到需要身份验证的网盘服务")
        special_handling['needs_auth'] = True
        special_handling['method_preference'] = ['stream']
        special_handling['storage_type'] = 'protected_drive'
    
    # GitHub releases/raw files
    elif 'github.com' in domain or 'githubusercontent.com' in domain:
        logger.info(f"🔍 检测到GitHub文件URL")
        special_handling['headers']['Accept'] = 'application/octet-stream'
        special_handling['storage_type'] = 'github'
    
    # CDN服务
    elif any(cdn in domain for cdn in ['jsdelivr.net', 'unpkg.com', 'cdnjs.cloudflare.com']):
        logger.info(f"🔍 检测到CDN服务URL")
        special_handling['method_preference'] = ['head', 'range', 'stream']
        special_handling['storage_type'] = 'cdn'
    
    # 检查URL中的特殊参数（如签名信息）
    if parsed.query:
        query_params = parse_qs(parsed.query)
        
        # 检查是否有临时签名参数
        signature_params = ['signature', 'sig', 'token', 'x-oss-signature', 'x-cos-signature', 'Expires']
        has_signature = any(param in query_params for param in signature_params)
        
        if has_signature:
            logger.info(f"🔐 检测到签名参数，可能为临时访问URL")
            special_handling['has_signature'] = True
            
            # 对于有签名的URL，通常需要完整保留参数
            special_handling['preserve_query'] = True
    
    logger.debug(f"🔧 URL处理结果: {special_handling}")
    return special_handling


def get_remote_file_size(url):
    """
    获取远程文件大小，支持各种类型的URL
    
    Args:
        url (str): 文件的URL地址
        
    Returns:
        int | None: 文件大小（字节），如果无法获取则返回None
    """
    import time
    import urllib3
    
    # 抑制SSL警告
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # 处理特殊URL类型
    special_config = _handle_special_urls(url)
    actual_url = special_config['url']
    
    logger.info(f"🔍 开始获取文件大小")
    logger.info(f"📎 原始URL: {url}")
    if actual_url != url:
        logger.info(f"🔄 转换后URL: {actual_url}")
    logger.info(f"⚙️ 特殊配置: {special_config}")
    
    if special_config['needs_auth']:
        logger.warning(f"🔒 URL可能需要身份验证，无法直接获取文件大小: {url}")
        return None
    
    # 配置请求头，模拟真实浏览器
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': '*/*',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0'
    }
    
    # 应用特殊配置的请求头
    headers.update(special_config['headers'])
    logger.debug(f"📋 请求头配置: {headers}")
    
    # 配置会话
    session = requests.Session()
    session.headers.update(headers)
    
    # 设置重试参数
    max_retries = 3
    retry_delay = 1
    timeout = 15
    
    # 根据特殊配置确定方法优先级
    method_preference = special_config.get('method_preference', ['head', 'range', 'stream'])
    logger.info(f"🎯 方法优先级: {method_preference}")
    
    for attempt in range(max_retries):
        try:
            logger.info(f"🚀 尝试获取文件大小 (第 {attempt + 1}/{max_retries} 次)")
            
            # 按照优先级尝试不同方法
            for method_idx, method in enumerate(method_preference):
                logger.info(f"🔧 使用方法 {method_idx + 1}/{len(method_preference)}: {method.upper()}")
                
                if method == 'head':
                    # 1. 首先尝试HEAD请求
                    try:
                        start_time = time.time()
                        response = session.head(
                            actual_url, 
                            allow_redirects=True, 
                            timeout=timeout,
                            verify=False  # 忽略SSL证书验证问题
                        )
                        request_time = time.time() - start_time
                        
                        logger.info(f"📡 HEAD请求完成 - 状态码: {response.status_code}, 耗时: {request_time:.2f}s")
                        logger.debug(f"📋 响应头: {dict(response.headers)}")
                        
                        # 检查响应状态码
                        if 200 <= response.status_code < 300:
                            content_length = response.headers.get('Content-Length')
                            if content_length and content_length.isdigit():
                                size = int(content_length)
                                size_mb = size / (1024 * 1024)
                                logger.success(f"✅ 通过HEAD请求获取到文件大小: {size:,} 字节 ({size_mb:.2f} MB)")
                                return size
                            
                            # 检查是否为网页内容
                            content_type = response.headers.get('Content-Type', '').lower()
                            if any(ct in content_type for ct in ['text/html', 'application/json', 'text/plain']):
                                logger.info(f"🌐 检测到网页内容，Content-Type: {content_type}")
                                continue  # 尝试下一个方法
                            else:
                                logger.warning(f"⚠️ HEAD请求无Content-Length头，Content-Type: {content_type}")
                        else:
                            logger.warning(f"⚠️ HEAD请求失败，状态码: {response.status_code}")
                                
                    except requests.exceptions.RequestException as e:
                        logger.warning(f"❌ HEAD请求异常: {type(e).__name__}: {e}")
                        continue
                
                elif method == 'range':
                    # 2. 使用Range请求获取部分内容
                    try:
                        range_headers = headers.copy()
                        range_headers['Range'] = 'bytes=0-0'
                        
                        start_time = time.time()
                        range_response = session.get(
                            actual_url, 
                            headers=range_headers,
                            timeout=timeout,
                            verify=False
                        )
                        request_time = time.time() - start_time
                        
                        logger.info(f"📡 Range请求完成 - 状态码: {range_response.status_code}, 耗时: {request_time:.2f}s")
                        logger.debug(f"📋 Range响应头: {dict(range_response.headers)}")
                        
                        if range_response.status_code == 206:  # Partial Content
                            content_range = range_response.headers.get('Content-Range')
                            if content_range:
                                logger.debug(f"📏 Content-Range: {content_range}")
                                # 解析 Content-Range: bytes 0-0/total_size
                                match = re.search(r'bytes \d+-\d+/(\d+)', content_range)
                                if match:
                                    total_size = int(match.group(1))
                                    size_mb = total_size / (1024 * 1024)
                                    logger.success(f"✅ 通过Range请求获取到文件大小: {total_size:,} 字节 ({size_mb:.2f} MB)")
                                    return total_size
                                else:
                                    logger.warning(f"⚠️ 无法解析Content-Range: {content_range}")
                            else:
                                logger.warning(f"⚠️ Range响应缺少Content-Range头")
                        else:
                            logger.warning(f"⚠️ Range请求失败，状态码: {range_response.status_code}")
                            
                    except requests.exceptions.RequestException as e:
                        logger.warning(f"❌ Range请求异常: {type(e).__name__}: {e}")
                        continue
                
                elif method == 'stream':
                    # 3. 使用流式GET请求
                    try:
                        start_time = time.time()
                        response = session.get(
                            actual_url, 
                            stream=True, 
                            allow_redirects=True, 
                            timeout=timeout,
                            verify=False
                        )
                        response.raise_for_status()
                        request_time = time.time() - start_time
                        
                        logger.info(f"📡 流式GET请求完成 - 状态码: {response.status_code}, 耗时: {request_time:.2f}s")
                        logger.debug(f"📋 GET响应头: {dict(response.headers)}")
                        
                        # 验证Content-Type，避免将错误页面当作文件
                        content_type = response.headers.get('Content-Type', '').lower()
                        logger.debug(f"📄 Content-Type: {content_type}")
                        
                        # 对于对象存储，检查是否返回了错误响应
                        storage_type = special_config.get('storage_type', 'generic')
                        if storage_type in ['aliyun_oss', 'huawei_obs', 'tencent_cos', 'aws_s3', 'google_gcs']:
                            # 检查是否为XML错误响应
                            if 'xml' in content_type or 'html' in content_type or 'json' in content_type:
                                logger.warning(f"⚠️ 对象存储返回了错误格式: {content_type}")
                                
                                # 读取部分内容进行验证
                                try:
                                    peek_size = min(1024, int(response.headers.get('Content-Length', 1024)))
                                    peek_content = b''
                                    for chunk in response.iter_content(chunk_size=peek_size):
                                        peek_content += chunk
                                        if len(peek_content) >= peek_size:
                                            break
                                    
                                    # 尝试解码查看内容
                                    try:
                                        text_content = peek_content.decode('utf-8', errors='ignore')
                                        logger.debug(f"📄 响应内容预览: {text_content[:200]}...")
                                        
                                        # 检查对象存储的典型错误响应
                                        error_patterns = [
                                            'accessdenied', 'access denied', 'nosuchkey', 'not found',
                                            'invalidrequest', 'signaturemismatch', 'requesttimeout',
                                            'error', 'exception', '<html', '<!doctype'
                                        ]
                                        
                                        text_lower = text_content.lower()
                                        if any(pattern in text_lower for pattern in error_patterns):
                                            logger.error(f"❌ 检测到对象存储错误响应")
                                            if 'access' in text_lower and 'denied' in text_lower:
                                                logger.error(f"🚫 访问被拒绝，可能是权限问题或签名过期")
                                            elif 'not found' in text_lower or 'nosuchkey' in text_lower:
                                                logger.error(f"🚫 文件不存在")
                                            else:
                                                logger.error(f"🚫 其他错误: {text_content[:100]}...")
                                            return None
                                            
                                    except UnicodeDecodeError:
                                        # 如果无法解码，可能是二进制文件，继续处理
                                        logger.debug(f"📄 无法解码响应内容，可能是二进制文件")
                                        pass
                                        
                                except Exception as e:
                                    logger.warning(f"⚠️ 内容验证失败: {e}")
                        
                        # 检查Content-Length头
                        content_length = response.headers.get('Content-Length')
                        if content_length and content_length.isdigit():
                            size = int(content_length)
                            size_mb = size / (1024 * 1024)
                            logger.success(f"✅ 通过GET请求获取到文件大小: {size:,} 字节 ({size_mb:.2f} MB)")
                            return size
                        
                        # 检查Transfer-Encoding
                        transfer_encoding = response.headers.get('Transfer-Encoding')
                        if transfer_encoding == 'chunked':
                            logger.warning(f"🔄 服务器使用分块传输，无法预先获取文件大小")
                            
                            # 对于分块传输，可以尝试下载一小部分来估算
                            if any(ct in content_type for ct in ['text/', 'application/json', 'application/xml']):
                                logger.info(f"📝 文本类型文件，尝试采样估算大小")
                                # 对于文本类型文件，下载一部分内容来估算
                                chunk_size = 8192
                                downloaded = 0
                                sample_chunks = []
                                
                                for chunk in response.iter_content(chunk_size=chunk_size):
                                    if chunk:
                                        sample_chunks.append(chunk)
                                        downloaded += len(chunk)
                                        if downloaded >= chunk_size * 5:  # 下载5个块作为样本
                                            break
                                
                                if sample_chunks:
                                    # 简单估算：假设平均块大小
                                    avg_chunk_size = downloaded / len(sample_chunks)
                                    logger.info(f"📊 基于样本估算，平均块大小: {avg_chunk_size:.0f} 字节，采样: {downloaded:,} 字节")
                                    # 返回None表示无法确定大小，但文件是可访问的
                                    return None
                            else:
                                logger.warning(f"❓ 无法确定分块传输文件的大小，Content-Type: {content_type}")
                                return None
                        
                        # 如果既没有Content-Length也不是chunked，尝试读取完整响应
                        logger.warning(f"❓ 既没有Content-Length也不是分块传输，无法确定文件大小")
                        return None
                        
                    except requests.exceptions.RequestException as e:
                        logger.error(f"❌ GET请求异常: {type(e).__name__}: {e}")
                        continue
            
            # 如果所有方法都失败了，进行重试
            if attempt < max_retries - 1:
                logger.warning(f"🔄 所有方法都失败，等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
                continue
            else:
                logger.error(f"💥 所有重试都失败了")
                return None
                     
        except Exception as e:
            logger.error(f"💥 获取文件大小时发生意外错误: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                logger.info(f"🔄 等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            else:
                return None
    
    logger.error(f"💥 无法获取文件大小: {url}")
    return None


def save_base64_file(base64_data: str,
                     save_dir: str,
                     file_type: str = None,
                     save_file_name: str = "download_file"):
    """
    保存 base64 文件，并确保扩展名正确
    - 优先通过 MIME 类型检测
    - 如果是 .bin/.unknown，尝试通过文件内容二次验证
    """
    try:
        file_data = base64.b64decode(base64_data)
        os.makedirs(save_dir, exist_ok=True)

        if not file_type:
            # 用户如果没有提供文件类型 进行guess 但不确保正确或无法识别
            kind = filetype.guess(file_data)
            if kind is not None:
                file_type = f".{kind.extension}"
            else:
                file_type = ".unknown"
        else:
            file_type = f".{file_type.lstrip('.')}"


        # 最终文件名
        final_name = f"{save_file_name}{file_type}"
        final_path = os.path.join(save_dir, final_name)
        with open(final_path, "wb") as f:
            f.write(file_data)

        return {
            "success": True,
            "file_path": final_path,
            "file_type": file_type,
            "message": "文件保存成功"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"保存失败: {str(e)}"
        }


def download_file(
        url,
        save_dir='.',
        file_type: str = None,
        save_file_name="download_file"
):
    """
    增强的文件下载函数，支持对象存储等特殊URL类型
    """
    temp_path = None
    try:
        # 检查是否为对象存储等特殊URL
        special_config = _handle_special_urls(url)
        
        # 使用特殊配置的URL和请求头
        actual_url = special_config['url']
        storage_type = special_config.get('storage_type', 'generic')
        
        # 基础请求头
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }
        
        # 应用特殊配置的请求头
        headers.update(special_config['headers'])
        
        print(f"📡 开始下载文件")
        print(f"🔗 URL: {actual_url}")
        print(f"🏷️ 存储类型: {storage_type}")
        print(f"📋 请求头: {headers}")
        
        # 对象存储特殊处理
        session = requests.Session()
        session.headers.update(headers)
        
        # 配置重试和超时
        timeout = 30
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                print(f"🚀 尝试下载 (第 {attempt + 1}/{max_retries} 次)")
                
                # 发送HTTP GET请求
                response = session.get(
                    actual_url, 
                    stream=True, 
                    allow_redirects=True,
                    timeout=timeout,
                    verify=False  # 对于内部或测试环境可能需要
                )
                
                print(f"📡 响应状态码: {response.status_code}")
                print(f"📋 响应头: {dict(response.headers)}")
                
                # 检查响应状态
                if response.status_code == 200:
                    break
                elif response.status_code == 403:
                    print(f"🚫 访问被拒绝 (403) - 可能需要身份验证或URL已过期")
                    if attempt == max_retries - 1:
                        raise Exception(f"访问被拒绝: HTTP {response.status_code}")
                elif response.status_code == 404:
                    print(f"🚫 文件不存在 (404)")
                    raise Exception(f"文件不存在: HTTP {response.status_code}")
                elif response.status_code >= 500:
                    print(f"🚫 服务器错误 ({response.status_code}) - 尝试重试")
                    if attempt == max_retries - 1:
                        raise Exception(f"服务器错误: HTTP {response.status_code}")
                else:
                    print(f"⚠️ 意外状态码: {response.status_code}")
                    if attempt == max_retries - 1:
                        raise Exception(f"请求失败: HTTP {response.status_code}")
                        
                # 等待重试
                import time
                wait_time = 2 ** attempt
                print(f"⏰ 等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                
            except requests.exceptions.Timeout:
                print(f"⏰ 请求超时")
                if attempt == max_retries - 1:
                    raise Exception("请求超时")
            except requests.exceptions.ConnectionError:
                print(f"🔌 连接错误")
                if attempt == max_retries - 1:
                    raise Exception("连接失败")
            except Exception as e:
                print(f"❌ 请求异常: {e}")
                if attempt == max_retries - 1:
                    raise

        # 验证Content-Type，避免下载到错误内容
        content_type = response.headers.get('Content-Type', '').lower()
        print(f"📄 Content-Type: {content_type}")
        
        # 修改后的错误检测逻辑
        if any(ct in content_type for ct in ['text/html', 'application/json', 'text/xml']):
            first_chunk = next(response.iter_content(chunk_size=1024), b'')
            try:
                text_content = first_chunk.decode('utf-8', errors='ignore')
                
                # 新增：检查是否为对象存储的错误响应
                is_object_storage_error = (
                    storage_type in ['aliyun_oss', 'huawei_obs', 'tencent_cos', 'aws_s3'] and
                    any(indicator in text_content.lower() 
                        for indicator in ['error', 'exception', 'accessdenied', 'nosuchkey'])
                )
                
                # 仅当确认是对象存储错误时才抛出异常
                if is_object_storage_error:
                    if 'access denied' in text_content.lower() or 'accessdenied' in text_content.lower():
                        raise Exception(f"对象存储访问被拒绝")
                    elif 'not found' in text_content.lower() or 'nosuchkey' in text_content.lower():
                        raise Exception(f"文件不存在")
                    else:
                        raise Exception(f"对象存储错误: {text_content[:100]}...")
                        
                # 否则记录警告但继续下载
                logger.warning(f"⚠️ 检测到文本类型内容，但继续下载: {content_type}")

            except UnicodeDecodeError:
                pass
            
            # 重新创建响应对象以包含已读取的chunk
            import io
            content = first_chunk + response.content
            response = type('MockResponse', (), {
                'headers': response.headers,
                'iter_content': lambda chunk_size=8192: [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
            })()

        # ------------------ 扩展名处理逻辑优化 ------------------
        # 标准化用户提供的扩展名（确保有前导点）
        if file_type:
            extension = f".{file_type.lstrip('.')}" if file_type else None
        else:
            # 自动检测扩展名的逻辑
            extension = None
            mime_type = content_type.split(';')[0].strip()

            # 来源1: Content-Type的MIME类型
            if mime_type and mime_type != 'application/octet-stream':
                extension = mimetypes.guess_extension(mime_type)
                logger.info(f"通过Content-Type检测到扩展名: {extension}")

            # 来源2: URL路径中的扩展名
            if not extension:
                path = urlparse(actual_url).path  # 使用处理后的URL
                url_extension = os.path.splitext(path)[1]
                extension = url_extension.lower() if url_extension else None
                logger.info(f"URL路径中的扩展名: {extension}")
                
            # 来源3: 文件内容检测（前两种方式失败时使用）
            if not extension:
                try:
                    # 创建临时文件
                    fd, temp_path = tempfile.mkstemp()
                    os.close(fd)

                    # 流式下载到临时文件
                    with open(temp_path, 'wb') as tmp_file:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                tmp_file.write(chunk)

                    # 使用magic检测MIME类型
                    try:
                        import magic
                        mime = magic.Magic(mime=True)
                        detected_mime = mime.from_file(temp_path)
                        extension = mimetypes.guess_extension(detected_mime) or '.unknown'
                        print(f"🔍 通过文件内容检测到扩展名: {extension}")
                    except (ImportError, Exception) as e:
                        print(f"⚠️ 文件内容检测失败: {e}")
                        extension = '.unknown'
                        
                except Exception as e:
                    print(f"❌ 创建临时文件失败: {e}")
                    extension = '.unknown'

        # 确保扩展名不为空
        if not extension:
            extension = '.unknown'

        # ------------------ 文件保存逻辑 ------------------
        # 生成最终路径
        save_path = os.path.join(save_dir, f"{save_file_name}{extension}")
        os.makedirs(save_dir, exist_ok=True)

        # 如果已下载到临时文件，直接移动；否则重新下载
        if temp_path and os.path.exists(temp_path):
            shutil.move(temp_path, save_path)
            print(f"✅ 文件已从临时位置移动到: {save_path}")
        else:
            print(f"💾 开始保存文件到: {save_path}")
            with open(save_path, 'wb') as f:
                if extension in ['.html', '.json', '.xml']:
                    f.write(content)
                    total_size = len(content) 
                else:
                    total_size = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            total_size += len(chunk)
                        
                print(f"💾 文件下载完成，总大小: {total_size} 字节")

        # 验证下载的文件
        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path)
            if file_size == 0:
                print(f"⚠️ 下载的文件大小为0，可能下载失败")
                os.remove(save_path)
                raise Exception("下载的文件为空")
            else:
                print(f"✅ 文件下载成功，大小: {file_size} 字节")

        return {
            "success": True,
            "file_path": save_path,
            "file_type": extension,
            "storage_type": storage_type,
            "message": "文件下载成功"
        }

    except requests.exceptions.RequestException as e:
        print(f"❌ 请求错误: {e}")
        return {"success": False, "error": str(e), "message": f"网络请求失败: {str(e)}"}
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return {"success": False, "error": str(e), "message": f"文件下载失败: {str(e)}"}
    finally:
        # 清理临时文件
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"🧹 清理临时文件: {temp_path}")
            except Exception as e:
                print(f"⚠️ 临时文件清理失败: {e}")


# 示例用法
if __name__ == "__main__":
    ...
    import requests
    import os

    # 文件URL
    file_url = "https://obsv3.coscoshipping-shdc-1.ex.cloud.coscoshipping.com/data-factory-test/fty/parsing/202508/%E5%B7%B2%E8%A7%A3%E6%9E%90_2019%20(50Mb)_1njfTQNhXfG.XLSX?AWSAccessKeyId=ZBS3SXIKXNFRMAFOQFVV&Expires=1756175086&Signature=JkWSkurJddZrRVqSXMYjDDOA7Qg%3D"

    # 本地保存路径
    save_path = "downloaded_file.xlsx"

    # 下载文件
    response = requests.get(file_url, stream=True)
    response.raise_for_status()

    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"文件已下载到: {os.path.abspath(save_path)}")
    # def download_file(
    #         url,
    #         save_dir='.',
    #         file_type: str = None,
    #         save_file_name="download_file"
    # ):


    # result = download_file(
    #     "https://arxiv.org/pdf/2501.12372",
    #     save_dir="output",
    #     save_file_name="download_file"
    # )
    # print(result, 111)
    # if result['success']:
    #     print(f"文件已保存至: {result['file_path']}")
    #     print(f"检测到的文件类型: {result['file_type']}")
    # else:
    #     print(f"下载失败: {result['message']}")


    # result = download_file(
    #     "https://arxiv.org/pdf/2501.12372",
    #     save_dir="output",
    #     save_file_name="download_file"
    # )
    # print(result, 111)
    # if result['success']:
    #     print(f"文件已保存至: {result['file_path']}")
    #     print(f"检测到的文件类型: {result['file_type']}")
    # else:
    #     print(f"下载失败: {result['message']}")