from flask import Flask, jsonify, request, Response, session, render_template, redirect, url_for, make_response
from flask_cors import CORS
import os
# from models import *
import pickle
from shutil import copy2
from time import sleep

app = Flask(__name__)
app.secret_key = os.urandom(24) 
CORS(app)

# Load all the models
# # Base model
# pprint("Loading Base model")
# model_base = Model_Base()
# model_base.load_weights()

# # Caption Self Attention model
# pprint("Loading Caption Self Attention model")
# model_caption_self_attention = Model_Caption_Self_Attention()
# model_caption_self_attention.load_weights()

# # Image Self Attention model
# pprint("Loading Image Self Attention model")
# model_image_self_attention = Model_Image_Self_Attention()
# model_image_self_attention.load_weights()

# # Caption_Image Self Attention model
# pprint("Loading Caption_Image Self Attention Model")
# model_caption_image_self_attention = Model_Caption_Image_Self_Attention()
# model_caption_image_self_attention.load_weights()



@app.route('/')
def home():
    return render_template('index.html')

'''
Get the caption and k value from user input
use either parameter passing or extract from POST array
@app.route("/getresults_model_1/<caption>", methods=['POST', 'GET'])
'''
@app.route("/base_model_query", methods=['POST', 'GET'])
def base_model_query():
    '''
    Takes the caption and k value
    Makes calls to backend function and predictions
    Returns: a list of image_names and scores
    
    '''
    caption = request.args['caption']
    mod_time = os.stat("result.pickle").st_mtime

    with open("queries.txt", 'w') as f:
        f.write(caption + "_1")

    
    while(os.stat("result.pickle").st_mtime == mod_time):
        pass
    
    sleep(1)
    with open('result.pickle', 'rb') as handle:
        results = pickle.load(handle)

    # results = model_base.predict(caption)
    print("Fetched Predictions")
    # #store the images to be rendered
    transfer_images_to_display(results)

    return render_template('gallery.html', predictions = results)

@app.route("/caption_self_attention_model_query", methods=['POST', 'GET'])
def caption_self_attention_model_query():
    '''
    Takes the caption and k value
    Makes calls to backend function and predictions
    Returns: a list of image_names and scores
    
    '''
    caption = request.args['caption']
    mod_time = os.stat("result.pickle").st_mtime

    with open("queries.txt", 'w') as f:
        f.write(caption + "_2")

    
    while(os.stat("result.pickle").st_mtime == mod_time):
        pass
    
    sleep(1)
    with open('result.pickle', 'rb') as handle:
        results = pickle.load(handle)

    # results = model_base.predict(caption)
    print("Fetched Predictions")
    # #store the images to be rendered
    transfer_images_to_display(results)

    return render_template('gallery.html', predictions = results)

@app.route("/image_self_attention_model_query", methods=['POST', 'GET'])
def image_self_attention_model_query():
    '''
    Takes the caption and k value
    Makes calls to backend function and predictions
    Returns: a list of image_names and scores
    
    '''
    caption = request.args['caption']
    mod_time = os.stat("result.pickle").st_mtime

    with open("queries.txt", 'w') as f:
        f.write(caption + "_3")

    
    while(os.stat("result.pickle").st_mtime == mod_time):
        pass
    
    sleep(1)
    with open('result.pickle', 'rb') as handle:
        results = pickle.load(handle)

    # results = model_base.predict(caption)
    print("Fetched Predictions")
    # #store the images to be rendered
    transfer_images_to_display(results)

    return render_template('gallery.html', predictions = results)


@app.route("/caption_image_self_attention_model_query", methods=['POST', 'GET'])
def caption_image_self_attention_model_query():
    '''
    Takes the caption and k value
    Makes calls to backend function and predictions
    Returns: a list of image_names and scores
    
    '''
    caption = request.args['caption']
    mod_time = os.stat("result.pickle").st_mtime

    with open("queries.txt", 'w') as f:
        f.write(caption + "_4")

    
    while(os.stat("result.pickle").st_mtime == mod_time):
        pass
    
    sleep(1)
    with open('result.pickle', 'rb') as handle:
        results = pickle.load(handle)

    # results = model_base.predict(caption)
    print("Fetched Predictions")
    # #store the images to be rendered
    transfer_images_to_display(results)

    return render_template('gallery.html', predictions = results)


def transfer_images_to_display(predictions):
    for _, img_path in predictions:
        print(img_path[1:])
        copy2(img_path[1:], "static/prediction_images/")
        
if __name__ == "__main__":
    app.run(host='0.0.0.0',port="5000", threaded=True)