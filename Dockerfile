FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LIVE_PAPER_CAPITAL_MODE=paper
ENV LIVE_PAPER_DASHBOARD_HOST=0.0.0.0
ENV LIVE_PAPER_DASHBOARD_PORT=8080

COPY requirements.txt requirements-live-paper.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-live-paper.txt

COPY . .

EXPOSE 8080

# Paper trading only — no broker order APIs are invoked by this entrypoint.
CMD ["python", "-m", "src.live_paper"]
