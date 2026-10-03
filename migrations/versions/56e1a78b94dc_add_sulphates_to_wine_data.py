"""add sulphates to wine data

Revision ID: 56e1a78b94dc
Revises: 92270fda58d9
Create Date: 2026-08-14 17:24:43.417931

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "56e1a78b94dc"
down_revision: str | Sequence[str] | None = "92270fda58d9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add sulphates column to wine_data."""

    op.add_column("wine_data", sa.Column("sulphates", sa.Float()))


def downgrade() -> None:
    """Remove sulphates column from wine_data."""

    op.drop_column("wine_data", "sulphates")
