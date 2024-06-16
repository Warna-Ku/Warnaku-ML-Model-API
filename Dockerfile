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

# Install specified Python packages
RUN pip install tensorflow==2.16.1 Keras Flask==2.2.2 Pillow==8.4.0 scikit-image==0.19.0 scikit-learn==0.24.2 requests joblib==1.1.0 Werkzeug==2.3.7

# Create a non-root user
RUN groupadd -r app && useradd -r -g app app

# Create application directory
RUN mkdir -p /opt/app

# Set working directory
WORKDIR /opt/app

# Copy application code
COPY . .

# Change ownership to non-root user
RUN chown -R app:app /opt/app
USER app

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application with Flask
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
