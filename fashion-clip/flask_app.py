import torch.nn.functional as F
from torch import Tensor
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import csv
import ast
import pandas as pd
import numpy as np
from PIL import Image
from IPython.display import display
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
import requests
import glob
from io import BytesIO
import re
from fashion_clip.fashion_clip import FashionCLIP
from flask import Flask,render_template,request

#Call the model
tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large')
model = AutoModel.from_pretrained('intfloat/e5-large')
fclip = FashionCLIP('fashion-clip')

#Normalize the e5 embeddings
def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

#Process e5 embeddings
def embed_query(data):
  batch_dict = tokenizer(data, max_length=512, padding=True, truncation=True, return_tensors='pt')
  output = model(**batch_dict)
  embeddings = average_pool(output.last_hidden_state, batch_dict['attention_mask'])
  embeddings = F.normalize(embeddings, p=2, dim=1)
  embeddings = embeddings.tolist()
  return embeddings

# Read embedding values from CSV
df = pd.read_csv('final_embeddings.csv')

#Read values as list
df['image_embeddings'] = df['image_embeddings'].apply(literal_eval)
# df['last_image_embeddings'] = df['last_image_embeddings'].apply(literal_eval)
df['text_embeddings_title'] = df['text_embeddings_title'].apply(literal_eval)
df['text_embeddings_colour'] = df['text_embeddings_colour'].apply(literal_eval)
df['text_embeddings_style'] = df['text_embeddings_style'].apply(literal_eval)
df['text_embeddings_features'] = df['text_embeddings_features'].apply(literal_eval)
df['text_embeddings_occasion'] = df['text_embeddings_occasion'].apply(literal_eval)
df['text_embeddings_category'] = df['text_embeddings_category'].apply(literal_eval)
# df['text_embeddings_text'] = df['text_embeddings_text'].apply(literal_eval)
# df['text_embeddings_new_title'] = df['text_embeddings_new_title'].apply(literal_eval)


images_embeddings = np.array(df['image_embeddings'])
images_embeddings = np.array([i for i in images_embeddings])
# last_images_embeddings = np.array(df['last_image_embeddings'])
# last_images_embeddings = np.array([i for i in last_images_embeddings])
text_embeddings_title = np.array(df['text_embeddings_title'])
text_embeddings_title = np.array([i for i in text_embeddings_title])
text_embeddings_colour = np.array(df['text_embeddings_colour'])
text_embeddings_colour = np.array([i for i in text_embeddings_colour])
text_embeddings_style = np.array(df['text_embeddings_style'])
text_embeddings_style = np.array([i for i in text_embeddings_style])
text_embeddings_features = np.array(df['text_embeddings_features'])
text_embeddings_features = np.array([i for i in text_embeddings_features])
text_embeddings_occasion = np.array(df['text_embeddings_occasion'])
text_embeddings_occasion = np.array([i for i in text_embeddings_occasion])
text_embeddings_category = np.array(df['text_embeddings_category'])
text_embeddings_category = np.array([i for i in text_embeddings_category])
# text_embeddings_text = np.array(df['text_embeddings_text'])
# text_embeddings_text = np.array([i for i in text_embeddings_text])
# text_embeddings_new_title = np.array(df['text_embeddings_new_title'])
# text_embeddings_new_title = np.array([i for i in text_embeddings_new_title])

#Asign weights
text = 0.5
image = 0.5
category = 5
title = 10
color = 4
occasion = 2
style = 2
features = 2


app = Flask(__name__)
app.template_folder = '/workspace'  
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        e5_query = embed_query(query)
        fclip_query = fclip.encode_text([query], 32)[0]
        fclip_query = np.array(fclip_query)
        fclip_query = fclip_query.reshape(1,-1)
       
        #Calculate scores with weights
        txt = ((title*text_embeddings_title) + 
       (color*text_embeddings_colour) +
       (category*text_embeddings_category) + 
       (occasion*text_embeddings_occasion)+
       (style*text_embeddings_style) + 
       (features*text_embeddings_features))
        e5_sim = [cosine_similarity(e5_query,[txt[i]])[0][0] for i in range(len(txt))]
        fclip_sim=[cosine_similarity(fclip_query,[images_embeddings[i]])[0][0] for i in range(len(txt))]
        score = [text*e5_sim[i] + image*fclip_sim[i] for i in range(len(txt))]

        #Get best 20 matches
        matched_indices = np.argsort(score)[-20:][::-1]
        matched_data = df.iloc[matched_indices]

        results = []
        count = 1

        #Send data to index.html for rendering in website
        for i, row in matched_data.iterrows():
            image_url = row['image_url']
            texts = str(str(count)+". "+row['text'])
            vals = score[i]
            results.append((image_url, texts, vals))
            count+=1

        return render_template('index.html', results=results)


    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug = True, host='0.0.0.0', port=7007)
