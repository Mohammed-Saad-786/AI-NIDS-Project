# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container
COPY . .

# Expose the port Streamlit runs on (default 8501)
EXPOSE 8501

# Command to run your app
# Note: server.port must match EXPOSE and the app_port in README.md
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
