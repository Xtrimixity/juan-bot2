# Dockerfile (for python 3.13 base)
FROM python:3.13-slim

# Create app dir
WORKDIR /app

# Install system deps (if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy code
COPY . .

# Use a non-root user (optional)
RUN useradd -m botuser
USER botuser

# Run the bot
CMD ["python", "bot.py"]
