# Use Python 3.9 base image
FROM python:3.9

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install TensorFlow first to avoid any potential conflicts
RUN pip install tensorflow==2.15.1

# Install other Python dependencies
RUN pip install Flask==2.0.2 Werkzeug==2.0.2 Pillow==8.4.0 scikit-image==0.19.0 scikit-learn==0.24.2

# Create application directory
RUN mkdir -p /opt/app

# Set working directory
WORKDIR /opt/app

# Copy the application code including the model
COPY ./models/UNet-ResNet34.keras /opt/app/models/UNet-ResNet34.keras
COPY . .

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD ["python", "app.py"]
