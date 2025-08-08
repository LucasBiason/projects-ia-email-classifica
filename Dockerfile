FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN apt-get update && apt-get install -y gcc libpq-dev
RUN pip install poetry
RUN poetry install

COPY . .

RUN chmod -R 777 /app

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]