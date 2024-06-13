FROM python:3.11-alpine

RUN mkdir -p /opt/app

WORKDIR /opt/app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# Define environment variable
ENV FLASK_APP=app.py

CMD ["flask", "run", "--host=0.0.0.0"] 