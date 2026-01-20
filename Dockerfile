# 1. Use a lightweight Python image
FROM python:3.9-slim

# 2. Create a folder inside the 'cloud box' for your app
WORKDIR /app

# 3. Basic tools needed for some ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy your files from your computer into the box
COPY . .

# 5. Install the libraries listed in your requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 6. Tell Render which port to open (Streamlit uses 8501)
EXPOSE 8501

# 7. THE ONLY LINE YOU CHANGE:
# Change 'your_filename.py' to match the file in this folder
CMD ["streamlit", "run", "fraud_app.py", ...]