FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY loci ./loci
COPY db ./db
COPY config.example.toml ./

RUN uv sync --frozen

ENV PATH="/app/.venv/bin:${PATH}"

ENTRYPOINT ["loci"]
CMD ["serve"]
