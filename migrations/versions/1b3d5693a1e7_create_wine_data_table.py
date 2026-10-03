"""create wine data table

Revision ID: 1b3d5693a1e7
Revises:
Create Date: 2026-08-13 23:58:25.423309

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "1b3d5693a1e7"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create wine_data table."""

    op.create_table(
        "wine_data",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("fixed_acidity", sa.Float()),
        sa.Column("volatile_acidity", sa.Float()),
        sa.Column("citric_acid", sa.Float()),
        sa.Column("residual_sugar", sa.Float()),
        sa.Column("chlorides", sa.Float()),
        sa.Column("free_sulfur_dioxide", sa.Float()),
        sa.Column("total_sulfur_dioxide", sa.Float()),
        sa.Column("density", sa.Float()),
        sa.Column("ph", sa.Float()),
        sa.Column("alcohol", sa.Float()),
        sa.Column("quality", sa.Integer()),
    )


def downgrade() -> None:
    """Drop wine_data table."""

    op.drop_table("wine_data")
