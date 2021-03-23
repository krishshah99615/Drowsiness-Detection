FROM python:3.8-slim-buster
COPY /webapp /app
WORKDIR /app
RUN apt-get update && apt-get -y install cmake protobuf-compiler
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit","run"]
CMD ["app.py"]