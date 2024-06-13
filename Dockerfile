FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev

# Create the application directory
RUN mkdir -p /opt/app

WORKDIR /opt/app

# Copy the requirements file
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8080

CMD ["python", "app.py"]