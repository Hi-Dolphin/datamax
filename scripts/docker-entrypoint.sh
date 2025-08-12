#!/bin/bash

# DataMax Docker Entrypoint Script
# This script handles container initialization and service startup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default configuration
DATAMAX_USER=${DATAMAX_USER:-datamax}
DATAMAX_UID=${DATAMAX_UID:-1000}
DATAMAX_GID=${DATAMAX_GID:-1000}
DATAMAX_HOME=${DATAMAX_HOME:-/app}
DATAMAX_DATA_DIR=${DATAMAX_DATA_DIR:-/app/data}
DATAMAX_LOG_DIR=${DATAMAX_LOG_DIR:-/app/logs}
DATAMAX_CONFIG_DIR=${DATAMAX_CONFIG_DIR:-/app/config}
DATAMAX_CACHE_DIR=${DATAMAX_CACHE_DIR:-/app/cache}

# Service configuration
DATAMAX_SERVICE=${DATAMAX_SERVICE:-web}
DATAMAX_HOST=${DATAMAX_HOST:-0.0.0.0}
DATAMAX_PORT=${DATAMAX_PORT:-8000}
DATAMAX_WORKERS=${DATAMAX_WORKERS:-4}
DATAMAX_LOG_LEVEL=${DATAMAX_LOG_LEVEL:-info}

# Database configuration
DATABASE_URL=${DATABASE_URL:-}
REDIS_URL=${REDIS_URL:-}
ELASTICSEARCH_URL=${ELASTICSEARCH_URL:-}

# Wait for services
WAIT_FOR_SERVICES=${WAIT_FOR_SERVICES:-true}
WAIT_TIMEOUT=${WAIT_TIMEOUT:-60}

# Development mode
DEV_MODE=${DEV_MODE:-false}
DEBUG=${DEBUG:-false}

# Function to wait for a service to be available
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local timeout=${4:-$WAIT_TIMEOUT}
    
    log_info "Waiting for $service_name at $host:$port..."
    
    local count=0
    while ! nc -z "$host" "$port" >/dev/null 2>&1; do
        if [ $count -ge $timeout ]; then
            log_error "Timeout waiting for $service_name at $host:$port"
            return 1
        fi
        count=$((count + 1))
        sleep 1
    done
    
    log_success "$service_name is available at $host:$port"
    return 0
}

# Function to parse database URL and wait for database
wait_for_database() {
    if [ -n "$DATABASE_URL" ]; then
        # Parse DATABASE_URL (format: postgresql://user:pass@host:port/db)
        local db_host=$(echo "$DATABASE_URL" | sed -n 's|.*://[^@]*@\([^:]*\):.*|\1|p')
        local db_port=$(echo "$DATABASE_URL" | sed -n 's|.*://[^@]*@[^:]*:\([0-9]*\)/.*|\1|p')
        
        if [ -n "$db_host" ] && [ -n "$db_port" ]; then
            wait_for_service "$db_host" "$db_port" "Database" || return 1
        else
            log_warning "Could not parse DATABASE_URL: $DATABASE_URL"
        fi
    fi
}

# Function to wait for Redis
wait_for_redis() {
    if [ -n "$REDIS_URL" ]; then
        # Parse REDIS_URL (format: redis://host:port)
        local redis_host=$(echo "$REDIS_URL" | sed -n 's|redis://\([^:]*\):.*|\1|p')
        local redis_port=$(echo "$REDIS_URL" | sed -n 's|redis://[^:]*:\([0-9]*\).*|\1|p')
        
        if [ -n "$redis_host" ] && [ -n "$redis_port" ]; then
            wait_for_service "$redis_host" "$redis_port" "Redis" || return 1
        else
            log_warning "Could not parse REDIS_URL: $REDIS_URL"
        fi
    fi
}

# Function to wait for Elasticsearch
wait_for_elasticsearch() {
    if [ -n "$ELASTICSEARCH_URL" ]; then
        # Parse ELASTICSEARCH_URL (format: http://host:port)
        local es_host=$(echo "$ELASTICSEARCH_URL" | sed -n 's|http://\([^:]*\):.*|\1|p')
        local es_port=$(echo "$ELASTICSEARCH_URL" | sed -n 's|http://[^:]*:\([0-9]*\).*|\1|p')
        
        if [ -n "$es_host" ] && [ -n "$es_port" ]; then
            wait_for_service "$es_host" "$es_port" "Elasticsearch" || return 1
        else
            log_warning "Could not parse ELASTICSEARCH_URL: $ELASTICSEARCH_URL"
        fi
    fi
}

# Function to create directories
create_directories() {
    log_info "Creating required directories..."
    
    local dirs=(
        "$DATAMAX_DATA_DIR"
        "$DATAMAX_LOG_DIR"
        "$DATAMAX_CONFIG_DIR"
        "$DATAMAX_CACHE_DIR"
        "$DATAMAX_DATA_DIR/raw"
        "$DATAMAX_DATA_DIR/processed"
        "$DATAMAX_DATA_DIR/exports"
    )
    
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        fi
    done
    
    # Set ownership if running as root
    if [ "$(id -u)" = "0" ]; then
        chown -R "$DATAMAX_UID:$DATAMAX_GID" "${dirs[@]}"
    fi
}

# Function to setup configuration
setup_configuration() {
    log_info "Setting up configuration..."
    
    local config_file="$DATAMAX_CONFIG_DIR/config.yaml"
    
    # Create default configuration if it doesn't exist
    if [ ! -f "$config_file" ]; then
        cat > "$config_file" << EOF
# DataMax Configuration
logging:
  level: ${DATAMAX_LOG_LEVEL^^}
  format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  handlers:
    - type: console
    - type: file
      filename: ${DATAMAX_LOG_DIR}/datamax.log
      max_bytes: 10485760  # 10MB
      backup_count: 5

server:
  host: ${DATAMAX_HOST}
  port: ${DATAMAX_PORT}
  workers: ${DATAMAX_WORKERS}
  debug: ${DEBUG}

crawler:
  max_concurrent: 10
  delay: 1.0
  timeout: 30.0
  retries: 3
  user_agent: 'DataMax/1.0'

storage:
  type: local
  path: ${DATAMAX_DATA_DIR}
  
parser:
  output_format: json
  clean_text: true
  extract_metadata: true

cache:
  type: file
  path: ${DATAMAX_CACHE_DIR}
  ttl: 3600

EOF

        # Add database configuration if available
        if [ -n "$DATABASE_URL" ]; then
            cat >> "$config_file" << EOF

database:
  url: ${DATABASE_URL}
  pool_size: 10
  max_overflow: 20
  pool_timeout: 30
  pool_recycle: 3600
EOF
        fi
        
        # Add Redis configuration if available
        if [ -n "$REDIS_URL" ]; then
            cat >> "$config_file" << EOF

redis:
  url: ${REDIS_URL}
  max_connections: 10
  socket_timeout: 5
  socket_connect_timeout: 5
EOF
        fi
        
        # Add Elasticsearch configuration if available
        if [ -n "$ELASTICSEARCH_URL" ]; then
            cat >> "$config_file" << EOF

elasticsearch:
  url: ${ELASTICSEARCH_URL}
  timeout: 30
  max_retries: 3
  retry_on_timeout: true
EOF
        fi
        
        log_success "Created configuration file: $config_file"
    else
        log_info "Configuration file already exists: $config_file"
    fi
    
    # Set ownership if running as root
    if [ "$(id -u)" = "0" ]; then
        chown "$DATAMAX_UID:$DATAMAX_GID" "$config_file"
    fi
}

# Function to run database migrations
run_migrations() {
    if [ -n "$DATABASE_URL" ]; then
        log_info "Running database migrations..."
        
        # Check if datamax has migration command
        if datamax --help | grep -q "migrate"; then
            datamax migrate --config "$DATAMAX_CONFIG_DIR/config.yaml" || {
                log_warning "Database migration failed, continuing..."
            }
        else
            log_info "No migration command available, skipping..."
        fi
    fi
}

# Function to validate environment
validate_environment() {
    log_info "Validating environment..."
    
    # Check if DataMax is installed
    if ! command -v datamax >/dev/null 2>&1; then
        log_error "DataMax command not found"
        exit 1
    fi
    
    # Check Python version
    local python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [ "$(echo "$python_version >= 3.10" | bc -l)" != "1" ]; then
        log_error "Python 3.10+ required, found $python_version"
        exit 1
    fi
    
    log_success "Environment validation passed"
}

# Function to switch to non-root user
switch_user() {
    if [ "$(id -u)" = "0" ]; then
        log_info "Switching to user $DATAMAX_USER (UID: $DATAMAX_UID, GID: $DATAMAX_GID)"
        
        # Create group if it doesn't exist
        if ! getent group "$DATAMAX_GID" >/dev/null 2>&1; then
            groupadd -g "$DATAMAX_GID" "$DATAMAX_USER"
        fi
        
        # Create user if it doesn't exist
        if ! getent passwd "$DATAMAX_UID" >/dev/null 2>&1; then
            useradd -u "$DATAMAX_UID" -g "$DATAMAX_GID" -d "$DATAMAX_HOME" -s /bin/bash "$DATAMAX_USER"
        fi
        
        # Execute command as non-root user
        exec gosu "$DATAMAX_USER" "$@"
    else
        # Already running as non-root user
        exec "$@"
    fi
}

# Function to start web server
start_web() {
    log_info "Starting DataMax web server..."
    
    local cmd="datamax serve"
    cmd="$cmd --host $DATAMAX_HOST"
    cmd="$cmd --port $DATAMAX_PORT"
    cmd="$cmd --workers $DATAMAX_WORKERS"
    cmd="$cmd --config $DATAMAX_CONFIG_DIR/config.yaml"
    
    if [ "$DEBUG" = "true" ]; then
        cmd="$cmd --debug"
    fi
    
    log_info "Executing: $cmd"
    exec $cmd
}

# Function to start worker
start_worker() {
    log_info "Starting DataMax worker..."
    
    local cmd="datamax worker"
    cmd="$cmd --config $DATAMAX_CONFIG_DIR/config.yaml"
    cmd="$cmd --log-level $DATAMAX_LOG_LEVEL"
    
    if [ -n "$WORKER_QUEUES" ]; then
        cmd="$cmd --queues $WORKER_QUEUES"
    fi
    
    if [ -n "$WORKER_CONCURRENCY" ]; then
        cmd="$cmd --concurrency $WORKER_CONCURRENCY"
    fi
    
    log_info "Executing: $cmd"
    exec $cmd
}

# Function to start scheduler
start_scheduler() {
    log_info "Starting DataMax scheduler..."
    
    local cmd="datamax scheduler"
    cmd="$cmd --config $DATAMAX_CONFIG_DIR/config.yaml"
    cmd="$cmd --log-level $DATAMAX_LOG_LEVEL"
    
    log_info "Executing: $cmd"
    exec $cmd
}

# Function to run crawl command
run_crawl() {
    log_info "Running DataMax crawl..."
    
    local cmd="datamax crawl"
    cmd="$cmd --config $DATAMAX_CONFIG_DIR/config.yaml"
    
    # Add any additional arguments passed to the script
    shift  # Remove 'crawl' from arguments
    cmd="$cmd $*"
    
    log_info "Executing: $cmd"
    exec $cmd
}

# Function to run parse command
run_parse() {
    log_info "Running DataMax parse..."
    
    local cmd="datamax parse"
    cmd="$cmd --config $DATAMAX_CONFIG_DIR/config.yaml"
    
    # Add any additional arguments passed to the script
    shift  # Remove 'parse' from arguments
    cmd="$cmd $*"
    
    log_info "Executing: $cmd"
    exec $cmd
}

# Function to run shell
run_shell() {
    log_info "Starting interactive shell..."
    exec /bin/bash
}

# Function to show help
show_help() {
    cat << EOF
DataMax Docker Entrypoint

Usage: $0 [COMMAND] [OPTIONS]

Commands:
  web         Start web server (default)
  worker      Start background worker
  scheduler   Start task scheduler
  crawl       Run crawl command
  parse       Run parse command
  shell       Start interactive shell
  help        Show this help message

Environment Variables:
  DATAMAX_SERVICE     Service to start (web|worker|scheduler) [default: web]
  DATAMAX_HOST        Server host [default: 0.0.0.0]
  DATAMAX_PORT        Server port [default: 8000]
  DATAMAX_WORKERS     Number of workers [default: 4]
  DATAMAX_LOG_LEVEL   Log level [default: info]
  
  DATABASE_URL        Database connection URL
  REDIS_URL          Redis connection URL
  ELASTICSEARCH_URL  Elasticsearch connection URL
  
  WAIT_FOR_SERVICES  Wait for external services [default: true]
  WAIT_TIMEOUT       Service wait timeout in seconds [default: 60]
  
  DEV_MODE           Enable development mode [default: false]
  DEBUG              Enable debug mode [default: false]

Examples:
  $0                    # Start web server
  $0 web                # Start web server explicitly
  $0 worker             # Start worker
  $0 crawl arxiv --query "machine learning" --max-results 10
  $0 parse --input /data --output /output
  $0 shell              # Interactive shell

EOF
}

# Main execution
main() {
    log_info "DataMax Docker Entrypoint Starting..."
    log_info "Service: ${DATAMAX_SERVICE}"
    log_info "User: ${DATAMAX_USER} (UID: ${DATAMAX_UID}, GID: ${DATAMAX_GID})"
    log_info "Home: ${DATAMAX_HOME}"
    
    # Validate environment
    validate_environment
    
    # Create required directories
    create_directories
    
    # Setup configuration
    setup_configuration
    
    # Wait for external services if enabled
    if [ "$WAIT_FOR_SERVICES" = "true" ]; then
        log_info "Waiting for external services..."
        wait_for_database || log_warning "Database not available"
        wait_for_redis || log_warning "Redis not available"
        wait_for_elasticsearch || log_warning "Elasticsearch not available"
    fi
    
    # Run database migrations
    run_migrations
    
    # Determine command to run
    local command="${1:-$DATAMAX_SERVICE}"
    
    case "$command" in
        web|serve)
            switch_user "$0" "_internal_web"
            ;;
        worker)
            switch_user "$0" "_internal_worker"
            ;;
        scheduler)
            switch_user "$0" "_internal_scheduler"
            ;;
        crawl)
            switch_user "$0" "_internal_crawl" "$@"
            ;;
        parse)
            switch_user "$0" "_internal_parse" "$@"
            ;;
        shell|bash)
            switch_user "$0" "_internal_shell"
            ;;
        help|--help|-h)
            show_help
            exit 0
            ;;
        _internal_web)
            start_web
            ;;
        _internal_worker)
            start_worker
            ;;
        _internal_scheduler)
            start_scheduler
            ;;
        _internal_crawl)
            run_crawl "$@"
            ;;
        _internal_parse)
            run_parse "$@"
            ;;
        _internal_shell)
            run_shell
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Handle signals
trap 'log_info "Received SIGTERM, shutting down..."; exit 0' TERM
trap 'log_info "Received SIGINT, shutting down..."; exit 0' INT

# Run main function
main "$@"