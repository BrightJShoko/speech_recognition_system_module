FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install necessary system dependencies
RUN apt-get update
RUN apt-get install libasound-dev libportaudio2 libportaudiocpp0 portaudio19-dev -y
RUN apt-get install gcc -y

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 5000
# Define environment variable
ENV FLASK_APP=app.py

# Run flask when the container launches
CMD ["flask", "run", "--host=0.0.0.0"]
