FROM python:3.11.7-slim

# Set the working directory in the container
WORKDIR /project-nyc-taxi-trip-duration

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container
COPY . .

# Command to run the train.py script
CMD ["python", "train.py"]