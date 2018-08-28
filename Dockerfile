# Use a base image that comes with NumPy and SciPy pre-installed
FROM publysher/alpine-scipy:1.0.0-numpy1.14.0-python3.6-alpine3.7
# Because of the image, our versions differ from those in the requirements.txt:
#   numpy==1.14.0 (instead of 1.13.1)
#   scipy==1.0.0 (instead of 0.19.1)


# Install sent2vec
RUN apk add --update git g++ make && \
    git clone https://github.com/epfml/sent2vec && \
    cd sent2vec && \
    git checkout f827d014a473aa22b2fef28d9e29211d50808d48 && \
    make && \
    apk del git make && \
    rm -rf /var/cache/apk/* && \
    pip install cython && \
    cd src && \
    python setup.py build_ext && \
    pip install .


# Install requirements
WORKDIR /app
ADD requirements.txt .

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt')"
# Download SpaCy model
RUN python -m spacy download en_core_web_sm

# Set the paths in config.ini
ADD config.ini.template config.ini
RUN sed -i '2 c\jar_path = /stanford-tagger/stanford-postagger.jar' config.ini && \
    sed -i '3 c\model_directory_path = /stanford-tagger/models/' config.ini && \
    sed -i '6 c\model_path = /sent2vec/pretrained_model.bin' config.ini

# Add actual source code
ADD embed_rank embed_rank/
ADD launch.py .

# Run python, optionally with launch.py as CMD
ENTRYPOINT ["python"]
CMD []
