# Use the official TensorFlow image as the base image
FROM tensorflow/tensorflow:2.16.1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install additional Python dependencies
RUN pip install --upgrade pip
RUN pip install Flask==2.0.2 Werkzeug==2.0.2 Pillow==8.4.0 scikit-image==0.19.0 scikit-learn==0.24.2

# Create the application directory
RUN mkdir -p /opt/app

# Set the working directory
WORKDIR /opt/app

# Copy the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD ["python", "app.py"]
