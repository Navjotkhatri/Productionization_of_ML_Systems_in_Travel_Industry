# Use the official Python 3.9 slim-buster image
FROM python:3.9-slim-buster

# Install system dependencies required by Python packages, including git
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container to /app
WORKDIR /app

# Copy all files from the current directory to /app in the container
COPY . /app

# Upgrade pip and install the dependencies listed in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 8000 to the outside world
EXPOSE 8000

# Set environment variable FLASK_APP to app.py
ENV FLASK_APP=app.py

# Command to run the Flask application
CMD ["python", "app.py"]
