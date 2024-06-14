FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install cython using pip
RUN python -m pip install --upgrade pip setuptools wheel cython

# Create the application directory
RUN mkdir -p /opt/app

WORKDIR /opt/app

# Copy the requirements file
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8080

CMD ["python", "app.py"]
