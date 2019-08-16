from models import *
import pickle
from time import sleep

# Load all the models

# Base model
pprint("Loading Base model")
model_base = Model_Base()
model_base.load_weights()

# Caption Self Attention model
pprint("Loading Caption Self Attention model")
model_caption_self_attention = Model_Caption_Self_Attention()
model_caption_self_attention.load_weights()

# Image Self Attention model
pprint("Loading Image Self Attention model")
model_image_self_attention = Model_Image_Self_Attention()
model_image_self_attention.load_weights()

# Caption_Image Self Attention model
pprint("Loading Caption_Image Self Attention Model")
model_caption_image_self_attention = Model_Caption_Image_Self_Attention()
model_caption_image_self_attention.load_weights()

mod_time = os.stat("queries.txt").st_mtime
while(1):
    if(os.stat("queries.txt").st_mtime != mod_time):
        # print()
        sleep(1)
        with open("queries.txt", 'r') as f:
            data = f.read().strip()

        data = data.split("_")

        if(data[1] == "1"):
            pprint("Base Model Prediction")
            res = model_base.predict(data[0])
            # print(res)
            with open('result.pickle', 'wb') as handle:
                pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if(data[1] == "2"):
            pprint("Caption Self Attention Model Prediction")
            res = model_caption_self_attention.predict(data[0])
            # print(res)
            with open('result.pickle', 'wb') as handle:
                pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if(data[1] == "3"):
            pprint("Image Self Attention Model Prediction")
            res = model_image_self_attention.predict(data[0])
            # print(res)
            with open('result.pickle', 'wb') as handle:
                pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if(data[1] == "4"):
            pprint("Caption-Image Self Attention Model Prediction")
            res = model_caption_image_self_attention.predict(data[0])
            # print(res)
            with open('result.pickle', 'wb') as handle:
                pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

        mod_time = os.stat("queries.txt").st_mtime
    