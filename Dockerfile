FROM python:3.11-slim

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
EXPOSE 5000

CMD ["python", "your_app.py"]