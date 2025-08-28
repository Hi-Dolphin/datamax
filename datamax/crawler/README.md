# DataMax Crawler Configuration

All configuration for the DataMax crawler module is now handled through environment variables, simplifying setup and deployment.

## Environment Variables

### Web Crawler Configuration
- `SEARCH_API_KEY` - API key for the web search API (required for web crawler)
- `WEB_SEARCH_API_URL` - URL for the web search API (default: https://api.bochaai.com/v1/web-search)
- `WEB_USER_AGENT` - User agent string for web requests (default: DataMax-Crawler/1.0)
- `WEB_TIMEOUT` - Timeout for web requests in seconds (default: 15)
- `WEB_MAX_RETRIES` - Maximum number of retry attempts (default: 2)
- `WEB_RATE_LIMIT` - Rate limit between requests in seconds (default: 0.5)

### ArXiv Crawler Configuration
- `ARXIV_BASE_URL` - Base URL for ArXiv API (default: https://arxiv.org/)
- `ARXIV_USER_AGENT` - User agent string for ArXiv requests (default: DataMax-Crawler/1.0)
- `ARXIV_TIMEOUT` - Timeout for ArXiv requests in seconds (default: 30)
- `ARXIV_MAX_RETRIES` - Maximum number of retry attempts (default: 3)
- `ARXIV_RATE_LIMIT` - Rate limit between requests in seconds (default: 1.0)

### Storage Configuration
- `STORAGE_DEFAULT_FORMAT` - Default storage format (json or yaml) (default: json)
- `STORAGE_OUTPUT_DIR` - Output directory for stored data (default: ./output)
- `STORAGE_CLOUD_ENABLED` - Enable cloud storage (true/false) (default: false)
- `STORAGE_CLOUD_PROVIDER` - Cloud storage provider (s3, gcs, azure) (default: s3)

### Logging Configuration
- `LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR) (default: INFO)
- `LOG_FILE` - Path to log file (optional)
- `LOG_ENABLE_JSON` - Enable JSON formatted logging (true/false) (default: false)
- `LOG_ENABLE_CONSOLE` - Enable console logging (true/false) (default: true)

## Usage Examples

### Setting Environment Variables (Linux/Mac)
```bash
export SEARCH_API_KEY="your-search-api-key"
export ARXIV_TIMEOUT=60
export STORAGE_OUTPUT_DIR="/path/to/output"
```

### Setting Environment Variables (Windows)
```cmd
set SEARCH_API_KEY=your-search-api-key
set ARXIV_TIMEOUT=60
set STORAGE_OUTPUT_DIR=C:\path\to\output
```

### Using with Docker
```dockerfile
ENV SEARCH_API_KEY=your-search-api-key
ENV ARXIV_TIMEOUT=60
ENV STORAGE_OUTPUT_DIR=/app/output
```