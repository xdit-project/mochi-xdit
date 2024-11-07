FROM thufeifeibear/xdit-dev:0.3.3

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/* && apt-get update

RUN apt-get install -y ffmpeg

# Install the project using pip
COPY . /app
WORKDIR /app
RUN pip install -e .

# Alternative approach using requirements.txt
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

