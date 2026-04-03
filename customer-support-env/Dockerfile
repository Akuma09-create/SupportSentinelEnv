# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Create a non-root user
RUN useradd -m -u 1000 user
USER user

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Copy the requirements file into the container
COPY --chown=user:user requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of the application code
COPY --chown=user:user . .

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Health check
HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

# Run app.py when the container launches
# Use the --host flag to expose the app on all network interfaces
# Add the user's local bin to the PATH
CMD ["/home/user/.local/bin/uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
