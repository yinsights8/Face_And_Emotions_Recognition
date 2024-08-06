FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Set the working directory in the container
WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
        python3.9 \
        python3-pip \
        build-essential \
        libffi-dev \
        libpq-dev \
        tzdata \
        libgl1-mesa-glx \
        v4l-utils \
        libgl1-mesa-glx \
        libglib2.0-0 \
        && ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
        dpkg-reconfigure --frontend noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/*

# Install OpenCV
RUN pip install opencv-python-headless

# Install your Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your application code
COPY . /app

# Expose port 8000 for the webserver
EXPOSE 8000

# Define the command to run the Flask app with Hypercorn
CMD ["hypercorn", "--bind", "0.0.0.0:8000", "ClientApp:Faceapp"]

# Optional: Add a health check (adjust the endpoint as needed)
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1