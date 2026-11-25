"""ARQ worker settings."""

from arq.connections import RedisSettings

from document_intelligence.config import settings
from document_intelligence.workers.extractor import process_document, shutdown, startup


class WorkerSettings:
    functions = [process_document]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = settings.worker_concurrency or 4
    job_timeout = settings.worker_timeout or 300
    keep_result = 86400  # 24 hours
    poll_delay = 0.5
    queue_name = settings.queue_name
