FROM python:3.11.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1
ENV UV_SYSTEM_PYTHON=1
ENV UV_LINK_MODE=copy
ENV ENVIRONMENT=demo

WORKDIR /app

COPY ./submodules /app/submodules

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

COPY . /app
COPY ./benchmarking /benchmarking

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --all-groups

ENTRYPOINT ["./start_env.sh"]