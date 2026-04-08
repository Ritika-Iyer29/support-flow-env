# Use python 3.10 for stability with openenv-core
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy all files (including the new /server folder)
COPY . .

# Install python dependencies
# Ensure httpx is included as it's used in your inference.py/app.py
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    openai \
    httpx \
    openenv-core

# Hugging Face Spaces uses port 7860
EXPOSE 7860

# CHANGE: Point to server.app:app because app.py is now inside the /server folder
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
