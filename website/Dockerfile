FROM python:3.12
WORKDIR /app
RUN mkdir -p data/uploads
RUN apt-get update && apt-get install -y build-essential libgomp1

COPY requirements_website.txt .
RUN pip install --no-cache-dir -r requirements_website.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "website:app"]