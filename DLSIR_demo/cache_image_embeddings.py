from models import *

pprint("Creating base model")
base_model = Model_Base()
pprint("Loading Weights of base model")
base_model.load_weights()
pprint("Cache image embeddings of base model")
base_model.cache_image_embeddings()

# pprint("Creating caption self attention model")
# caption_self_attention_model = Model_Caption_Self_Attention()
# pprint("Loading Weights of caption self attention model")
# caption_self_attention_model.load_weights()
# pprint("Cache image embeddings of caption self attention model")
# caption_self_attention_model.cache_image_embeddings()

# pprint("Creating image self attention model")
# image_self_attention_model = Model_Image_Self_Attention()
# pprint("Loading Weights of image self attention model")
# image_self_attention_model.load_weights()
# pprint("Cache image embeddings of image self attention model")
# image_self_attention_model.cache_image_embeddings()

# pprint("Creating caption-image self attention model")
# caption_img_self_attention_model = Model_Caption_Image_Self_Attention()
# pprint("Loading Weights of caption-image self attention model")
# caption_img_self_attention_model.load_weights()
# pprint("Cache image embeddings of caption-image self attention model")
# caption_img_self_attention_model.cache_image_embeddings()
