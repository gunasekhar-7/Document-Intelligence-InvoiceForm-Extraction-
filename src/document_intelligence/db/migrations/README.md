# Database Migrations

This directory contains Alembic migration scripts for the document_intelligence database.

## Commands

```bash
# Create new migration (auto-generated from model changes)
alembic revision --autogenerate -m "description of changes"

# Create empty migration
alembic revision -m "description of changes"

# Apply all migrations
alembic upgrade head

# Apply specific migration
alembic upgrade <revision>

# Rollback one migration
alembic downgrade -1

# Show current revision
alembic current

# Show migration history
alembic history

# Show migration SQL (dry run)
alembic upgrade head --sql
```

## Migration Files

Migration files are stored in `versions/` with format:
`YYYY_MM_DD_HHMMSS_revision_slug.py`

## Best Practices

1. Always review auto-generated migrations before applying
2. Test migrations on a copy of production data first
3. Use descriptive revision messages
4. Keep migrations small and focused
5. Never modify applied migrations - create new ones instead