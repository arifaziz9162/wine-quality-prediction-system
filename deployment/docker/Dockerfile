# This image is only used to run database migrations.
# Migration tools are kept separate from the application serving container.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install only the packages needed to connect to PostgreSQL
# and run the Alembic migrations.
RUN pip install --no-cache-dir \
    alembic==1.18.5 \
    SQLAlchemy==2.0.35 \
    psycopg2-binary==2.9.9 \
    python-dotenv==1.0.1

# Copy the Alembic configuration and migration files.
COPY alembic.ini .
COPY migrations ./migrations

# Run all pending database migrations.
CMD ["alembic", "upgrade", "head"]
