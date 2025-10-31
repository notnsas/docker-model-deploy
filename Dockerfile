# Base image containing Python runtime
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the application files from the host to the container
COPY . /app

# Install the dependencies listed in requirements.txt
RUN pip install -r requirements.txt

# Define the command to run the Flask app when the container starts
CMD ["python", "model_deployment.py"]