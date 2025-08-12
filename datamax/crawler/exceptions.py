"""Crawler Exception Classes

Defines custom exceptions for the DataMax crawler module.
"""


class CrawlerException(Exception):
    """Base exception class for all crawler-related errors.
    
    This is the base class for all exceptions that can be raised
    by crawler implementations.
    """
    pass


class NetworkException(CrawlerException):
    """Exception raised for network-related errors.
    
    This includes connection timeouts, DNS resolution failures,
    HTTP errors, and other network-related issues.
    """
    pass


class ParseException(CrawlerException):
    """Exception raised for data parsing errors.
    
    This includes malformed HTML/XML, unexpected data formats,
    and other parsing-related issues.
    """
    pass


class RateLimitException(CrawlerException):
    """Exception raised when rate limits are exceeded.
    
    This is raised when the crawler encounters rate limiting
    from the target website or API.
    """
    pass


class AuthenticationException(CrawlerException):
    """Exception raised for authentication-related errors.
    
    This includes invalid credentials, expired tokens,
    and other authentication issues.
    """
    pass


class ConfigurationException(CrawlerException):
    """Exception raised for configuration-related errors.
    
    This includes missing required configuration parameters,
    invalid configuration values, and other setup issues.
    """
    pass