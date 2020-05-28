FROM tensorflow/tensorflow:1.13.2-py3
WORKDIR /app
ADD . /app
# Set Flask ENV
ENV FLASK_APP app.py
ENV FLASK_RUN_HOST 0.0.0.0
EXPOSE 5000
RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt
CMD ["flask", "run"]

