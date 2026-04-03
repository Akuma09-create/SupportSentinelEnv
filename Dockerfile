FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose the port
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "customer-support-env.app:app", "--host", "0.0.0.0", "--port", "7860"]
