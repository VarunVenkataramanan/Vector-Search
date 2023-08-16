FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
WORKDIR /workspace
# RUN pip install -r requirements.txt
RUN pip install notebook boto3==1.10.50 appdirs==1.4.4 pandas==1.3.5 pyarrow==7.0.0 matplotlib==3.5.1 python-dotenv==0.19.2 annoy==1.17.0 transformers==4.26.1 ipyplot==1.1.1 datasets==2.10.0 validators scikit-learn Flask transformers[sentencepiece] Pillow ipython	
# CMD ["/bin/bash"]
#CMD ["jupyter","notebook","--ip","0.0.0.0","--no-browser","--allow-root"]
EXPOSE 7007
CMD ["python","flask_app.py"]
