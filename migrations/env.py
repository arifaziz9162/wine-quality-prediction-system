import os
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, pool

# Load environment variables
load_dotenv()

# Alembic config
config = context.config

# Config logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Create database URL
db_url = URL.create(
    drivername="postgresql+psycopg2",
    username=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    database=os.getenv("DB_NAME"),
)

# Migrations are managed manually, so no ORM metadata is needed
target_metadata = None


def run_migrations_offline() -> None:
    """Run migrations without a database connection."""

    context.configure(
        url=db_url.render_as_string(hide_password=False),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations using a database connection."""

    # Create database engine
    connectable = create_engine(db_url, poolclass=pool.NullPool)

    # Run migrations
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()

    # Close engine
    connectable.dispose()


# Run migrations in the selected mode
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
