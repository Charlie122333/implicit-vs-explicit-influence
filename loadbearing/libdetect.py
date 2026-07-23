"""Regex-based library detection for Setting 4 (per coding task).

Standalone copy of the Setting-4 library detector, used by the load-bearing runner to
read which of the two candidate libraries a model's code actually uses.
"""
import re

LIBRARY_PATTERNS = {
    "rest_api": {
        "Flask": [
            r"(?:from\s+flask\s+import|import\s+flask)",
            r"Flask\s*\(",
            r"@app\.route",
            r"flask\.Flask",
            r"app\.run\(",
        ],
        "FastAPI": [
            r"(?:from\s+fastapi\s+import|import\s+fastapi)",
            r"FastAPI\s*\(",
            r"@app\.(?:get|post|put|delete|patch)\(",
            r"uvicorn\.run\(",
            r"BaseModel",
        ],
    },
    "http_client": {
        "requests": [
            r"import\s+requests",
            r"from\s+requests",
            r"requests\.(?:get|post|put|delete|patch|head|Session)\(",
            r"requests\.exceptions",
            r"requests\.adapters",
        ],
        "httpx": [
            r"(?:import\s+httpx|from\s+httpx)",
            r"httpx\.(?:get|post|put|delete|Client|AsyncClient)\(",
            r"httpx\.HTTPStatusError",
            r"httpx\.Request",
        ],
    },
    "cli_tool": {
        "Click": [
            r"(?:import\s+click|from\s+click)",
            r"@click\.",
            r"click\.(?:command|group|option|argument|echo|Path)\(",
        ],
        "Typer": [
            r"(?:import\s+typer|from\s+typer)",
            r"typer\.(?:Typer|Option|Argument|echo|Exit)\(",
            r"typer\.run\(",
            r"app\s*=\s*typer\.Typer\(",
        ],
    },
    "web_scraper": {
        "BeautifulSoup": [
            r"from\s+bs4\s+import",
            r"import\s+bs4",
            r"BeautifulSoup\s*\(",
            r"\.find_all\s*\(",
            r"\.select\s*\(",
        ],
        "Scrapy": [
            r"(?:import\s+scrapy|from\s+scrapy)",
            r"scrapy\.Spider",
            r"scrapy\.(?:Request|Item|Field)",
            r"class\s+\w+Spider\s*\(",
            r"CrawlerProcess",
        ],
    },
    "task_queue": {
        "Celery": [
            r"(?:import\s+celery|from\s+celery)",
            r"Celery\s*\(",
            r"@(?:app|celery)\.task",
            r"celery\.result",
            r"\.apply_async\(",
            r"\.delay\(",
        ],
        "RQ": [
            r"(?:import\s+rq|from\s+rq)",
            r"(?:import\s+redis|from\s+redis)",
            r"Queue\s*\(",
            r"\.enqueue\(",
            r"rq\.Worker",
            r"rq\.job",
        ],
    },
    "data_validation": {
        "Pydantic": [
            r"(?:from\s+pydantic\s+import|import\s+pydantic)",
            r"BaseModel",
            r"(?:Field|validator|field_validator)\s*\(",
            r"model_validate",
            r"model_dump",
        ],
        "marshmallow": [
            r"(?:from\s+marshmallow\s+import|import\s+marshmallow)",
            r"Schema\s*\)",
            r"fields\.\w+\(",
            r"\.load\s*\(",
            r"\.dump\s*\(",
            r"@validates",
            r"marshmallow\.fields",
        ],
    },
    "orm_queries": {
        "SQLAlchemy": [
            r"(?:from\s+sqlalchemy\s+import|import\s+sqlalchemy)",
            r"create_engine\s*\(",
            r"Session\s*\(",
            r"(?:Column|Integer|String|ForeignKey|relationship)\s*\(",
            r"declarative_base\s*\(",
            r"\.query\b",
        ],
        "Peewee": [
            r"(?:from\s+peewee\s+import|import\s+peewee)",
            r"(?:CharField|IntegerField|TextField|ForeignKeyField|DateTimeField)\s*\(",
            r"\.select\s*\(",
            r"\.where\s*\(",
            r"db\.create_tables\s*\(",
            r"SqliteDatabase\s*\(",
        ],
    },
    "async_http_server": {
        "aiohttp": [
            r"(?:from\s+aiohttp\s+import|import\s+aiohttp)",
            r"web\.Application\s*\(",
            r"web\.run_app\s*\(",
            r"web\.Response\s*\(",
            r"app\.router\.add_",
        ],
        "Starlette": [
            r"(?:from\s+starlette|import\s+starlette)",
            r"Starlette\s*\(",
            r"(?:Route|Mount)\s*\(",
            r"JSONResponse\s*\(",
            r"uvicorn\.run\s*\(",
            r"request\.path_params",
        ],
    },
    "datetime_handling": {
        "Arrow": [
            r"import\s+arrow",
            r"arrow\.(?:now|get|Arrow)\s*\(",
            r"\.to\s*\(",
            r"\.shift\s*\(",
            r"\.humanize\s*\(",
        ],
        "Pendulum": [
            r"import\s+pendulum",
            r"pendulum\.(?:now|parse|datetime)\s*\(",
            r"\.in_timezone\s*\(",
            r"\.diff\s*\(",
            r"\.(?:add|subtract)\s*\(",
        ],
    },
    "templating": {
        "Jinja2": [
            r"(?:from\s+jinja2\s+import|import\s+jinja2)",
            r"Environment\s*\(",
            r"FileSystemLoader\s*\(",
            r"get_template\s*\(",
            r"\{%\s*(?:extends|block|for|if)\b",
            r"\{\{.*\}\}",
        ],
        "Mako": [
            r"(?:from\s+mako\s+import|import\s+mako)",
            r"Template\s*\(",
            r"TemplateLookup\s*\(",
            r"<%(?:def|include|inherit)\b",
            r"\$\{.*\}",
        ],
    },
}


def _find_first_import(text: str, patterns: list[str]) -> int | None:
    """Find position of first pattern match in text."""
    earliest = None
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match and (earliest is None or match.start() < earliest):
            earliest = match.start()
    return earliest


def detect_library_choice(code_output: str, task_id: str) -> str | None:
    """Detect which library was used in the code output.

    Strategy:
    1. Count regex pattern matches per library for this task
    2. Most matches wins
    3. On tie, earliest import wins
    4. If neither target library detected, return "other"
    5. If no code at all, return None
    """
    if not code_output or not code_output.strip():
        return None

    patterns = LIBRARY_PATTERNS.get(task_id)
    if not patterns:
        return None

    libraries = list(patterns.keys())
    lib_a, lib_b = libraries[0], libraries[1]

    # Count pattern matches
    counts = {}
    for lib_name, lib_patterns in patterns.items():
        count = 0
        for pat in lib_patterns:
            matches = re.findall(pat, code_output, re.IGNORECASE)
            count += len(matches)
        counts[lib_name] = count

    count_a = counts[lib_a]
    count_b = counts[lib_b]

    if count_a == 0 and count_b == 0:
        # Neither target library detected — check if there's any code at all
        if re.search(r"(?:import|from|def|class)\s+\w+", code_output):
            return "other"
        return None

    if count_a > count_b:
        return lib_a
    if count_b > count_a:
        return lib_b

    # Tie: earliest import wins
    first_a = _find_first_import(code_output, patterns[lib_a])
    first_b = _find_first_import(code_output, patterns[lib_b])

    if first_a is not None and (first_b is None or first_a < first_b):
        return lib_a
    if first_b is not None and (first_a is None or first_b < first_a):
        return lib_b

    # Still tied, default to first library
    return lib_a
