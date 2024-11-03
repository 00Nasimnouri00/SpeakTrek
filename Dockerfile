# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Install ffmpeg for audio processing with librosa and pydub
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the rest of your application code into the container
COPY . .

# Environment variables for AssemblyAI API Key
# Replace YOUR_ASSEMBLYAI_API_KEY with your actual AssemblyAI API Key, or set it in a .env file
# ENV ASSEMBLYAI_API_KEY YOUR_ASSEMBLYAI_API_KEY

# Run the main application (replace app.py with your actual script name)
CMD ["python", "main2.py"]
