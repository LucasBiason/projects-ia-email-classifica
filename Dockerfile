FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /app

COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install -y gcc libpq-dev
RUN pip install -r requirements.txt

COPY . .

RUN chmod -R 777 /app

ENTRYPOINT ["/entrypoint.sh"]