"""create predictions table

Revision ID: 92270fda58d9
Revises: 1b3d5693a1e7
Create Date: 2026-08-14 00:01:39.356231

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "92270fda58d9"
down_revision: str | Sequence[str] | None = "1b3d5693a1e7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create predictions table."""

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("fixed_acidity", sa.Float()),
        sa.Column("volatile_acidity", sa.Float()),
        sa.Column("citric_acid", sa.Float()),
        sa.Column("residual_sugar", sa.Float()),
        sa.Column("chlorides", sa.Float()),
        sa.Column("free_sulfur_dioxide", sa.Float),
        sa.Column("total_sulfur_dioxide", sa.Float()),
        sa.Column("density", sa.Float()),
        sa.Column("ph", sa.Float()),
        sa.Column("sulphates", sa.Float()),
        sa.Column("alcohol", sa.Float()),
        sa.Column("predicted_quality", sa.Float()),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP")
        ),
    )


def downgrade() -> None:
    """Drop predictions table."""

    op.drop_table("predictions")
