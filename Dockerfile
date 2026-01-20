# 1. Use a lightweight Python image
FROM python:3.9-slim

# 2. Create a folder inside the 'cloud box' for your app
WORKDIR /app

# 3. Basic tools needed for ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy your files from your computer into the box
COPY . .

# 5. Install the libraries listed in your requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 6. Tell Render which port to open
EXPOSE 8501

# 7. Run the app (MAKE SURE THIS FILENAME IS CORRECT)
CMD ["streamlit", "run", "3_fraud_detection.py", "--server.port=8501", "--server.address=0.0.0.0"]
