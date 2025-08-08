# Use official Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Expose default port (Render sets $PORT at runtime)
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run Streamlit on the provided $PORT (Render, Railway, etc.)
CMD ["sh", "-c", "streamlit run app.py --server.port $PORT --server.address 0.0.0.0"]