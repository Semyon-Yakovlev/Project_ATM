FROM python:3.11-buster as builder

RUN pip install poetry==1.4.2

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /dashboard

COPY poetry.lock pyproject.toml ./

RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --without dev --no-root

FROM python:3.11-slim-buster as runtime

ENV VIRTUAL_ENV=/dashboard/.venv \
    PATH="/dashboard/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY . /dashboard

ENTRYPOINT ["streamlit", "run", "/dashboard/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]