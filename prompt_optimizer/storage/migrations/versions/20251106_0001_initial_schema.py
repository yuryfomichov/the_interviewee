"""Initial schema for prompt optimizer

Revision ID: 0001
Revises:
Create Date: 2025-11-06

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '0001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial schema with all tables and indexes."""
    # Create optimization_runs table
    op.create_table(
        'optimization_runs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('task_description', sa.Text(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('champion_prompt_id', sa.String(), nullable=True),
        sa.Column('total_tests_run', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(), nullable=False, server_default='running'),
        sa.Column('total_time_seconds', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # Create prompts table
    op.create_table(
        'prompts',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('run_id', sa.Integer(), nullable=False),
        sa.Column('prompt_text', sa.Text(), nullable=False),
        sa.Column('stage', sa.String(), nullable=False),
        sa.Column('strategy', sa.String(), nullable=True),
        sa.Column('average_score', sa.Float(), nullable=True),
        sa.Column('quick_score', sa.Float(), nullable=True),
        sa.Column('rigorous_score', sa.Float(), nullable=True),
        sa.Column('iteration', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('track_id', sa.Integer(), nullable=True),
        sa.Column('parent_prompt_id', sa.String(), nullable=True),
        sa.Column('is_original_system_prompt', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['run_id'], ['optimization_runs.id'], ),
        sa.ForeignKeyConstraint(['parent_prompt_id'], ['prompts.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create test_cases table
    op.create_table(
        'test_cases',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('run_id', sa.Integer(), nullable=False),
        sa.Column('input_message', sa.Text(), nullable=False),
        sa.Column('expected_behavior', sa.Text(), nullable=False),
        sa.Column('category', sa.String(), nullable=False),
        sa.Column('stage', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['run_id'], ['optimization_runs.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create evaluations table
    op.create_table(
        'evaluations',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('run_id', sa.Integer(), nullable=False),
        sa.Column('test_case_id', sa.String(), nullable=False),
        sa.Column('prompt_id', sa.String(), nullable=False),
        sa.Column('model_response', sa.Text(), nullable=False),
        sa.Column('functionality', sa.Integer(), nullable=False),
        sa.Column('safety', sa.Integer(), nullable=False),
        sa.Column('consistency', sa.Integer(), nullable=False),
        sa.Column('edge_case_handling', sa.Integer(), nullable=False),
        sa.Column('reasoning', sa.Text(), nullable=False),
        sa.Column('overall_score', sa.Float(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['run_id'], ['optimization_runs.id'], ),
        sa.ForeignKeyConstraint(['test_case_id'], ['test_cases.id'], ),
        sa.ForeignKeyConstraint(['prompt_id'], ['prompts.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create weakness_analyses table
    op.create_table(
        'weakness_analyses',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('prompt_id', sa.String(), nullable=False),
        sa.Column('iteration', sa.Integer(), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('failed_test_ids', sa.Text(), nullable=False),
        sa.Column('failed_test_descriptions', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['prompt_id'], ['prompts.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes for performance
    op.create_index('idx_prompts_run_stage', 'prompts', ['run_id', 'stage'])
    op.create_index('idx_prompts_score', 'prompts', ['average_score'], postgresql_ops={'average_score': 'DESC'})
    op.create_index('idx_prompts_track', 'prompts', ['run_id', 'track_id'])
    op.create_index('idx_evaluations_run', 'evaluations', ['run_id'])
    op.create_index('idx_evaluations_prompt', 'evaluations', ['prompt_id'])
    op.create_index('idx_evaluations_test', 'evaluations', ['test_case_id'])
    op.create_index('idx_test_cases_run_stage', 'test_cases', ['run_id', 'stage'])
    op.create_index('idx_weakness_analyses_prompt', 'weakness_analyses', ['prompt_id'])

    # Add foreign key for champion_prompt_id (must be added after prompts table exists)
    op.create_foreign_key(
        'fk_optimization_runs_champion',
        'optimization_runs',
        'prompts',
        ['champion_prompt_id'],
        ['id']
    )


def downgrade() -> None:
    """Drop all tables and indexes."""
    op.drop_constraint('fk_optimization_runs_champion', 'optimization_runs', type_='foreignkey')

    op.drop_index('idx_weakness_analyses_prompt', table_name='weakness_analyses')
    op.drop_index('idx_test_cases_run_stage', table_name='test_cases')
    op.drop_index('idx_evaluations_test', table_name='evaluations')
    op.drop_index('idx_evaluations_prompt', table_name='evaluations')
    op.drop_index('idx_evaluations_run', table_name='evaluations')
    op.drop_index('idx_prompts_track', table_name='prompts')
    op.drop_index('idx_prompts_score', table_name='prompts')
    op.drop_index('idx_prompts_run_stage', table_name='prompts')

    op.drop_table('weakness_analyses')
    op.drop_table('evaluations')
    op.drop_table('test_cases')
    op.drop_table('prompts')
    op.drop_table('optimization_runs')
