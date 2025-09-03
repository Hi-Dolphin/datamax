#!/usr/bin/env python3
"""
DataMax Database Migration Script

This script handles database schema migrations for DataMax.
It supports SQLite, PostgreSQL, and MySQL databases.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import sqlalchemy as sa
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import Engine
    from sqlalchemy.exc import SQLAlchemyError
except ImportError:
    print("SQLAlchemy is required for database migrations.")
    print("Install it with: pip install sqlalchemy")
    sys.exit(1)

try:
    import yaml
except ImportError:
    yaml = None

# Migration configuration
MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations"
MIGRATIONS_TABLE = "datamax_migrations"

# Default database URLs for different environments
DEFAULT_DATABASE_URLS = {
    'development': 'sqlite:///./datamax-dev.db',
    'test': 'sqlite:///./datamax-test.db',
    'production': None  # Must be provided
}


class MigrationError(Exception):
    """Custom exception for migration errors."""
    pass


class Migration:
    """Represents a single database migration."""
    
    def __init__(self, version: str, name: str, up_sql: str, down_sql: str = None):
        self.version = version
        self.name = name
        self.up_sql = up_sql
        self.down_sql = down_sql
        self.timestamp = datetime.now()
    
    def __str__(self):
        return f"Migration {self.version}: {self.name}"
    
    def __repr__(self):
        return f"Migration(version='{self.version}', name='{self.name}')"


class MigrationManager:
    """Manages database migrations for DataMax."""
    
    def __init__(self, database_url: str, migrations_dir: Path = None, verbose: bool = False):
        self.database_url = database_url
        self.migrations_dir = migrations_dir or MIGRATIONS_DIR
        self.verbose = verbose
        
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create database engine
        try:
            self.engine = create_engine(database_url, echo=verbose)
            self.logger.info(f"Connected to database: {self._mask_url(database_url)}")
        except Exception as e:
            raise MigrationError(f"Failed to connect to database: {e}")
        
        # Ensure migrations directory exists
        self.migrations_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize migrations table
        self._init_migrations_table()
    
    def _mask_url(self, url: str) -> str:
        """Mask sensitive information in database URL."""
        if '://' in url:
            scheme, rest = url.split('://', 1)
            if '@' in rest:
                credentials, host_part = rest.split('@', 1)
                if ':' in credentials:
                    user, _ = credentials.split(':', 1)
                    return f"{scheme}://{user}:***@{host_part}"
                else:
                    return f"{scheme}://***@{host_part}"
        return url
    
    def _init_migrations_table(self):
        """Initialize the migrations tracking table."""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {MIGRATIONS_TABLE} (
            version VARCHAR(50) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            checksum VARCHAR(64)
        )
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(create_table_sql))
                conn.commit()
            self.logger.debug(f"Migrations table '{MIGRATIONS_TABLE}' initialized")
        except Exception as e:
            raise MigrationError(f"Failed to initialize migrations table: {e}")
    
    def _get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(f"SELECT version FROM {MIGRATIONS_TABLE} ORDER BY version")
                )
                return [row[0] for row in result]
        except Exception as e:
            raise MigrationError(f"Failed to get applied migrations: {e}")
    
    def _mark_migration_applied(self, migration: Migration):
        """Mark a migration as applied."""
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text(f"""
                    INSERT INTO {MIGRATIONS_TABLE} (version, name, applied_at)
                    VALUES (:version, :name, :applied_at)
                    """),
                    {
                        'version': migration.version,
                        'name': migration.name,
                        'applied_at': migration.timestamp
                    }
                )
                conn.commit()
        except Exception as e:
            raise MigrationError(f"Failed to mark migration as applied: {e}")
    
    def _mark_migration_reverted(self, version: str):
        """Mark a migration as reverted (remove from applied migrations)."""
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text(f"DELETE FROM {MIGRATIONS_TABLE} WHERE version = :version"),
                    {'version': version}
                )
                conn.commit()
        except Exception as e:
            raise MigrationError(f"Failed to mark migration as reverted: {e}")
    
    def _load_migrations(self) -> List[Migration]:
        """Load all migration files from the migrations directory."""
        migrations = []
        
        # Look for SQL files
        for sql_file in sorted(self.migrations_dir.glob('*.sql')):
            version = sql_file.stem
            name = version.replace('_', ' ').title()
            
            with open(sql_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split up and down migrations if present
            if '-- DOWN' in content:
                parts = content.split('-- DOWN', 1)
                up_sql = parts[0].replace('-- UP', '').strip()
                down_sql = parts[1].strip()
            else:
                up_sql = content.replace('-- UP', '').strip()
                down_sql = None
            
            migrations.append(Migration(version, name, up_sql, down_sql))
        
        # Look for Python files
        for py_file in sorted(self.migrations_dir.glob('*.py')):
            if py_file.name.startswith('__'):
                continue
                
            version = py_file.stem
            name = version.replace('_', ' ').title()
            
            # Import the migration module
            spec = importlib.util.spec_from_file_location(version, py_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get up and down functions
            up_func = getattr(module, 'up', None)
            down_func = getattr(module, 'down', None)
            
            if up_func:
                migrations.append(Migration(version, name, up_func, down_func))
        
        return sorted(migrations, key=lambda m: m.version)
    
    def _execute_sql(self, sql: str, migration_name: str = None):
        """Execute SQL statements."""
        if not sql.strip():
            return
        
        try:
            with self.engine.connect() as conn:
                # Split SQL into individual statements
                statements = [stmt.strip() for stmt in sql.split(';') if stmt.strip()]
                
                for stmt in statements:
                    if stmt:
                        self.logger.debug(f"Executing: {stmt[:100]}...")
                        conn.execute(text(stmt))
                
                conn.commit()
                
        except Exception as e:
            error_msg = f"Failed to execute SQL"
            if migration_name:
                error_msg += f" for migration '{migration_name}'"
            error_msg += f": {e}"
            raise MigrationError(error_msg)
    
    def _execute_python(self, func, migration_name: str = None):
        """Execute Python migration function."""
        try:
            func(self.engine)
        except Exception as e:
            error_msg = f"Failed to execute Python migration"
            if migration_name:
                error_msg += f" '{migration_name}'"
            error_msg += f": {e}"
            raise MigrationError(error_msg)
    
    def create_migration(self, name: str, sql: bool = True) -> Path:
        """Create a new migration file."""
        # Generate version based on timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version = f"{timestamp}_{name.lower().replace(' ', '_')}"
        
        if sql:
            filename = f"{version}.sql"
            template = f"""
-- Migration: {name}
-- Created: {datetime.now().isoformat()}

-- UP
-- Add your migration SQL here


-- DOWN
-- Add your rollback SQL here (optional)

"""
        else:
            filename = f"{version}.py"
            template = f"""
# Migration: {name}
# Created: {datetime.now().isoformat()}

import sqlalchemy as sa
from sqlalchemy import text


def up(engine):
    \"\"\"Apply the migration.\"\"\"
    with engine.connect() as conn:
        # Add your migration code here
        pass


def down(engine):
    \"\"\"Rollback the migration.\"\"\"
    with engine.connect() as conn:
        # Add your rollback code here
        pass
"""
        
        migration_file = self.migrations_dir / filename
        
        if migration_file.exists():
            raise MigrationError(f"Migration file already exists: {migration_file}")
        
        with open(migration_file, 'w', encoding='utf-8') as f:
            f.write(template.strip())
        
        self.logger.info(f"Created migration: {migration_file}")
        return migration_file
    
    def migrate(self, target_version: str = None) -> int:
        """Apply pending migrations."""
        migrations = self._load_migrations()
        applied = set(self._get_applied_migrations())
        
        # Filter migrations to apply
        pending = [m for m in migrations if m.version not in applied]
        
        if target_version:
            pending = [m for m in pending if m.version <= target_version]
        
        if not pending:
            self.logger.info("No pending migrations")
            return 0
        
        self.logger.info(f"Applying {len(pending)} migration(s)...")
        
        applied_count = 0
        for migration in pending:
            try:
                self.logger.info(f"Applying migration: {migration}")
                
                if isinstance(migration.up_sql, str):
                    self._execute_sql(migration.up_sql, migration.name)
                else:
                    self._execute_python(migration.up_sql, migration.name)
                
                self._mark_migration_applied(migration)
                applied_count += 1
                
                self.logger.info(f"✓ Applied migration: {migration.version}")
                
            except Exception as e:
                self.logger.error(f"✗ Failed to apply migration {migration.version}: {e}")
                raise
        
        self.logger.info(f"Successfully applied {applied_count} migration(s)")
        return applied_count
    
    def rollback(self, target_version: str = None, steps: int = 1) -> int:
        """Rollback migrations."""
        migrations = {m.version: m for m in self._load_migrations()}
        applied = self._get_applied_migrations()
        
        if not applied:
            self.logger.info("No migrations to rollback")
            return 0
        
        # Determine which migrations to rollback
        if target_version:
            to_rollback = [v for v in reversed(applied) if v > target_version]
        else:
            to_rollback = applied[-steps:] if steps <= len(applied) else applied
            to_rollback = list(reversed(to_rollback))
        
        if not to_rollback:
            self.logger.info("No migrations to rollback")
            return 0
        
        self.logger.info(f"Rolling back {len(to_rollback)} migration(s)...")
        
        rolled_back = 0
        for version in to_rollback:
            migration = migrations.get(version)
            if not migration:
                self.logger.warning(f"Migration file not found for version: {version}")
                self._mark_migration_reverted(version)
                continue
            
            if not migration.down_sql:
                raise MigrationError(f"No rollback defined for migration: {version}")
            
            try:
                self.logger.info(f"Rolling back migration: {migration}")
                
                if isinstance(migration.down_sql, str):
                    self._execute_sql(migration.down_sql, migration.name)
                else:
                    self._execute_python(migration.down_sql, migration.name)
                
                self._mark_migration_reverted(version)
                rolled_back += 1
                
                self.logger.info(f"✓ Rolled back migration: {version}")
                
            except Exception as e:
                self.logger.error(f"✗ Failed to rollback migration {version}: {e}")
                raise
        
        self.logger.info(f"Successfully rolled back {rolled_back} migration(s)")
        return rolled_back
    
    def status(self) -> Dict:
        """Get migration status."""
        migrations = self._load_migrations()
        applied = set(self._get_applied_migrations())
        
        pending = [m for m in migrations if m.version not in applied]
        applied_migrations = [m for m in migrations if m.version in applied]
        
        return {
            'total_migrations': len(migrations),
            'applied_count': len(applied_migrations),
            'pending_count': len(pending),
            'applied_migrations': [m.version for m in applied_migrations],
            'pending_migrations': [m.version for m in pending],
            'database_url': self._mask_url(self.database_url)
        }
    
    def reset(self) -> int:
        """Reset database by rolling back all migrations."""
        applied = self._get_applied_migrations()
        if applied:
            return self.rollback(steps=len(applied))
        return 0


def load_config(config_file: Path = None) -> Dict:
    """Load configuration from file."""
    if not config_file or not config_file.exists():
        return {}
    
    if not yaml:
        return {}
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def get_database_url(args) -> str:
    """Get database URL from various sources."""
    # Command line argument takes precedence
    if args.database_url:
        return args.database_url
    
    # Environment variable
    if os.getenv('DATABASE_URL'):
        return os.getenv('DATABASE_URL')
    
    # Configuration file
    if args.config:
        config = load_config(Path(args.config))
        if 'database' in config and 'url' in config['database']:
            return config['database']['url']
    
    # Default for environment
    env = args.environment or os.getenv('DATAMAX_ENV', 'development')
    default_url = DEFAULT_DATABASE_URLS.get(env)
    
    if not default_url:
        raise MigrationError(
            f"No database URL configured for environment '{env}'. "
            "Please provide --database-url or set DATABASE_URL environment variable."
        )
    
    return default_url


def main():
    """Main entry point for migration script."""
    parser = argparse.ArgumentParser(
        description="DataMax Database Migration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s migrate                    # Apply all pending migrations
  %(prog)s migrate --target 20231201  # Migrate to specific version
  %(prog)s rollback                   # Rollback last migration
  %(prog)s rollback --steps 3         # Rollback last 3 migrations
  %(prog)s rollback --target 20231201 # Rollback to specific version
  %(prog)s status                     # Show migration status
  %(prog)s create "Add users table"   # Create new migration
  %(prog)s reset                      # Reset database (rollback all)

Environment Variables:
  DATABASE_URL     Database connection URL
  DATAMAX_ENV      Environment name (development, test, production)
        """
    )
    
    parser.add_argument(
        '--database-url',
        help='Database connection URL'
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--environment', '-e',
        choices=['development', 'test', 'production'],
        help='Environment name'
    )
    
    parser.add_argument(
        '--migrations-dir',
        type=Path,
        default=MIGRATIONS_DIR,
        help=f'Migrations directory (default: {MIGRATIONS_DIR})'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', help='Apply pending migrations')
    migrate_parser.add_argument(
        '--target',
        help='Target migration version'
    )
    
    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback migrations')
    rollback_group = rollback_parser.add_mutually_exclusive_group()
    rollback_group.add_argument(
        '--target',
        help='Target migration version to rollback to'
    )
    rollback_group.add_argument(
        '--steps',
        type=int,
        default=1,
        help='Number of migrations to rollback (default: 1)'
    )
    
    # Status command
    subparsers.add_parser('status', help='Show migration status')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create new migration')
    create_parser.add_argument(
        'name',
        help='Migration name'
    )
    create_parser.add_argument(
        '--python',
        action='store_true',
        help='Create Python migration instead of SQL'
    )
    
    # Reset command
    subparsers.add_parser('reset', help='Reset database (rollback all migrations)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        # Get database URL
        database_url = get_database_url(args)
        
        # Create migration manager
        manager = MigrationManager(
            database_url=database_url,
            migrations_dir=args.migrations_dir,
            verbose=args.verbose
        )
        
        # Execute command
        if args.command == 'migrate':
            count = manager.migrate(target_version=args.target)
            return 0 if count >= 0 else 1
            
        elif args.command == 'rollback':
            if args.target:
                count = manager.rollback(target_version=args.target)
            else:
                count = manager.rollback(steps=args.steps)
            return 0 if count >= 0 else 1
            
        elif args.command == 'status':
            status = manager.status()
            print(f"Database: {status['database_url']}")
            print(f"Total migrations: {status['total_migrations']}")
            print(f"Applied: {status['applied_count']}")
            print(f"Pending: {status['pending_count']}")
            
            if status['applied_migrations']:
                print("\nApplied migrations:")
                for version in status['applied_migrations']:
                    print(f"  ✓ {version}")
            
            if status['pending_migrations']:
                print("\nPending migrations:")
                for version in status['pending_migrations']:
                    print(f"  • {version}")
            
            return 0
            
        elif args.command == 'create':
            migration_file = manager.create_migration(
                name=args.name,
                sql=not args.python
            )
            print(f"Created migration: {migration_file}")
            return 0
            
        elif args.command == 'reset':
            count = manager.reset()
            return 0 if count >= 0 else 1
            
        else:
            parser.print_help()
            return 1
            
    except MigrationError as e:
        print(f"Migration error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())