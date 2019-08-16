from Model_definitions import *

from sklearn.utils import shuffle
import tensorflow as tf


annotation_file = 'annotations/captions_train2014.json'

with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# storing the captions and the image name in vectors
all_captions = []# only has the captions
all_img_name_vector = []# only has the paths

for annot in annotations['annotations']:
    #have to see if each image has 1:5 relation with the captions and have to incorporate that somehow!
    caption = '<start> ' + annot['caption'] + ' <end>'
    image_id = annot['image_id']
    full_coco_image_path = 'train2014/' + 'COCO_train2014_' + '%012d.jpg' % (image_id)
    
    all_img_name_vector.append(full_coco_image_path)
    all_captions.append(caption)

# shuffling the captions and image_names together
# setting a random state
train_captions, img_name_vector = shuffle(all_captions,
                                          all_img_name_vector,
                                          random_state=1)



# The steps above is a general process of dealing with text processing
# choosing the top 18000 words from the vocabulary
top_k = 18000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, 
                                                  oov_token="<unk>", 
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)
tokenizer.word_index['<pad>'] = 0

pprint("Loading Test data")
img_name_test = np.load('MS_COCO/img_name_test.npy')
cap_test = np.load('MS_COCO/cap_test.npy')

vocab_size = len(tokenizer.word_index)

params_set = {
    "embed_dim":300,
    "split_ratio":0.33,
    "max_len":200,
    "vocab_size":10000,
    "trainable_param":"False",
    "option":2,
    "workers":3,
    "window":1,
}
#embedding matrix
def prepare_embedding_matrix(model_wv, tokenizer_dict):
  '''
    takes the word vector model and the vocabulary and build an embedding matrix
  '''
  embedding_matrix = [0] * len(tokenizer_dict)
  # embedding_matrix = np.zeros(len(tokenizer_dict))
  for word, index in tokenizer_dict.items():
    embedding_matrix[index] = model_wv[word]
  
  return np.array(embedding_matrix)

pprint("Preparing Embedding Matrix") 
if(not(os.path.exists("embedding_matrix.npy"))):
  # prepare the embedding matrix
  embedding_matrix = prepare_embedding_matrix(model_wv,tokenizer.word_index)
  np.save("embedding_matrix", embedding_matrix)
else:
  embedding_matrix = np.load("embedding_matrix.npy")  
  
print("Size of embedding matrix = ", embedding_matrix.shape)

def caption_to_post_padding(caption):
  caption = '<start> ' + caption + ' <end>'
  tokens = caption.split()
  pad_amount = max_length - len(tokens)
  padding = ['<pad>'] * pad_amount
  tokens.extend(padding)
  return tokens

def caption_to_embedding(caption):
  def get_word_index(x):
    try:
      return tokenizer.word_index[x]
    except:
      return 0 #out of vocabulary have to look into this later
  
  tokens = caption_to_post_padding(caption)
  index = list(map(lambda x:get_word_index(x),tokens))
  return np.array(index)

def get_top_k_images(model, imgs_embs_path, caption, k):
  cap_inp = caption_to_embedding(caption)
  test_model_cap, _ = model.get_test_models_from_weights()
  cap_emb = test_model_cap.predict(np.array([cap_inp]))

  imgs_embs = np.load(imgs_embs_path)
  similarity_scores = []
  for img_emb in imgs_embs:
    score=order_violations(cap_emb, img_emb)
    similarity_scores.append(score)
  combined = list(zip(similarity_scores, img_name_test))
  combined = list(map(list, combined))
  topk = sorted(combined, key=lambda x: x[0])[:k]
  # print(topk)

  for i in range(len(topk)):
    topk[i][0] = round((1 - topk[i][0])*100, 2)

  return topk

class Model_Base:
  def __init__(self):
    self.image_input = Input(shape=model_config['image_input_shape'], name='image_input')
    X = Dense(2048)(self.image_input)
    X = Dense(1024)(X)
    X = Flatten()(X)
    X = Dense(model_config['output_dim'])(X)
    self.emb_image = Lambda(lambda x: l2norm(x))(X)   #the final embedding representation of the image
    self.cap_input = Input(shape=(model_config['max_cap_length'],), dtype='float32', name='cap_input')
    X = Masking(mask_value=0,input_shape=(model_config['max_cap_length'], model_config['output_dim']))(self.cap_input)#(X)
    X = Embedding(output_dim=model_config['dim_word'], input_dim=len(tokenizer.word_index), input_length=model_config['max_cap_length'], weights=[embedding_matrix])(X) #(cap_input)
    _,for_state_h,for_state_c,bac_state_h,bac_state_c=Bidirectional(lstm(model_config['output_dim']))(X)
    mergedstates = concatenate([for_state_h,for_state_c,bac_state_h,bac_state_c])
    X = Dense(2048)(mergedstates)
    X = Dense(model_config['output_dim'])(X)
    self.emb_cap = Lambda(lambda x: l2norm(x))(X)   #the final embedding representation of the catption
    merged = concatenate([self.emb_cap, self.emb_image])
    self.model = Model(inputs=[self.cap_input, self.image_input], outputs=[merged])

    self.model.compile(optimizer=model_config['optimizer'], loss=contrastive_loss)
  
  def load_weights(self):
    with open('DLSIR_model_weights/inception+fasttext+bidir/model_base.json', 'r') as json_file:
      loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("DLSIR_model_weights/inception+fasttext+bidir/model_weights_base.h5")
    self.model = loaded_model
    self.model.compile(optimizer=model_config['optimizer'], loss=contrastive_loss)
    print("Loaded model from disk")

  def get_test_models_from_weights(self):
    weights = self.model.get_weights()
  
    im_w = weights[7:11] + weights[15:17]
    emb_w = weights[0:7] + weights[11:15]

    test_model_im = Model(inputs=self.image_input, outputs=self.emb_image)
    test_model_im.set_weights(im_w)
    test_model_im.compile(optimizer='adam', loss="mse")#contrastive_loss)
    test_model_cap = Model(inputs=self.cap_input, outputs=self.emb_cap)
    test_model_cap.set_weights(emb_w)
    test_model_cap.compile(optimizer='adam', loss="mse")#contrastive_loss)

    return test_model_cap, test_model_im
    
  def cache_image_embeddings(self):
    _, test_model_im = self.get_test_models_from_weights()

    img_name_test = np.load('MS_COCO/img_name_test.npy')
    
    img_name_test = list(map(lambda x: x[1:], img_name_test))
    all_test_images = img_name_test[:6000]

    all_test_image_features = list(map(lambda img_name:np.load(img_name +'.npy'), all_test_images))
    imgs_embs = test_model_im.predict(np.array(all_test_image_features))

    np.save('cached_image_embs/model_base/imgs_embs', imgs_embs)
    
  def predict(self, caption, k=9):
    return get_top_k_images(self, 'cached_image_embs/model_base/imgs_embs.npy', caption, k)

class Model_Caption_Self_Attention:
  def __init__(self):
    self.image_input = Input(shape=model_config['image_input_shape'], name='image_input')
    X = Dense(2048, activation='relu')(self.image_input)
    X = Dense(1024, activation='relu')(X)
    X = Flatten()(X)
    X = Dense(model_config['output_dim'], activation='relu')(X)
    self.emb_image = Lambda(lambda x: l2norm(x))(X)

    self.cap_input = Input(shape=(model_config['max_cap_length'],), dtype='float32', name='cap_input')
    X = Masking(mask_value=0,input_shape=(model_config['max_cap_length'],))(self.cap_input)
    X = Embedding(output_dim=model_config['dim_word'], input_dim=len(tokenizer.word_index), input_length=model_config['max_cap_length'], weights=np.array([embedding_matrix]))(X)
    X = Dense(2048, activation='relu')(X)
    X = Dense(1024, activation='relu')(X)

    X = Position_Embedding()(X) 
    X = Attention(model_config["num_heads"], model_config["units"])([X, X, X])
    X = GlobalAveragePooling1D()(X)

    X = Flatten()(X)
    X = Dense(2048, activation='relu')(X)
    X = Dense(model_config['output_dim'], activation='relu')(X)


    self.emb_cap = Lambda(lambda x: l2norm(x))(X)
    merged = concatenate([self.emb_cap, self.emb_image])

    self.model = Model(inputs=[self.cap_input, self.image_input], outputs=[merged])
    self.model.compile(optimizer=tf.train.AdamOptimizer(), loss=contrastive_loss)

  def load_weights(self):
    with open('DLSIR_model_weights/inception+fasttext+caption_self_attention/model_caption_self_attention.json', 'r') as json_file:
      loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json, custom_objects={'Position_Embedding': Position_Embedding, 'Attention': Attention})
    
    loaded_model.load_weights("DLSIR_model_weights/inception+fasttext+caption_self_attention/model_weights_caption_self_attention.h5")
    self.model = loaded_model
    #recompile this loaded model
    self.model.compile(optimizer=tf.train.AdamOptimizer(), loss=contrastive_loss)
    print("Loaded model from disk")

  def get_test_models_from_weights(self):
    weights = self.model.get_weights()
  
    im_w = weights[8:12] + weights[16:18] #12:18
    c_w = weights[0:8] + weights[12:16]


    test_model_im = Model(inputs=self.image_input, outputs=self.emb_image)
    test_model_im.set_weights(im_w)
    test_model_im.compile(optimizer='adam', loss="mse")#contrastive_loss)
    test_model_cap = Model(inputs=self.cap_input, outputs=self.emb_cap)
    test_model_cap.set_weights(c_w)
    test_model_cap.compile(optimizer='adam', loss="mse")#contrastive_loss)
    return test_model_cap,test_model_im

  def cache_image_embeddings(self):
    _, test_model_im = self.get_test_models_from_weights()

    img_name_test = np.load('MS_COCO/img_name_test.npy')
    
    img_name_test = list(map(lambda x: x[1:], img_name_test))
    all_test_images = img_name_test[:6000]

    all_test_image_features = list(map(lambda img_name:np.load(img_name +'.npy'), all_test_images))
    imgs_embs = test_model_im.predict(np.array(all_test_image_features))

    np.save('cached_image_embs/model_caption_self_attention/imgs_embs', imgs_embs)

  def predict(self, caption, k=9):
    return get_top_k_images(self, 'cached_image_embs/model_caption_self_attention/imgs_embs.npy', caption, k)

class Model_Image_Self_Attention:
  def __init__(self):
    self.image_input = Input(shape=model_config['image_input_shape'], name='image_input')
    X = Dense(2048, activation='relu')(self.image_input)
    X = Dense(1024, activation='relu')(X)
    X = Position_Embedding()(X) 
    X = Attention(model_config["num_heads"], model_config["units"])([X, X, X])
    X = GlobalAveragePooling1D()(X)
    X = Flatten()(X)
    X = Dense(2048,activation='relu')(X)
    X = Dense(model_config['output_dim'])(X)
    self.emb_image = Lambda(lambda x: l2norm(x))(X)
    self.cap_input = Input(shape=(model_config['max_cap_length'],), dtype='float32', name='cap_input')
    X = Masking(mask_value=0,input_shape=(model_config['max_cap_length'],))(self.cap_input)
    X = Embedding(output_dim=model_config['dim_word'], input_dim=len(tokenizer.word_index), input_length=model_config['max_cap_length'], weights=[embedding_matrix])(X)
    _,for_state_h,for_state_c,bac_state_h,bac_state_c=Bidirectional(lstm(model_config['output_dim']))(X)
    mergedstates = concatenate([for_state_h,for_state_c,bac_state_h,bac_state_c])
    X = Dense(2048,activation='relu')(mergedstates)
    X= Dense(model_config['output_dim'])(X)
    self.emb_cap = Lambda(lambda x: l2norm(x))(X)   #the final embedding representation of the catption
    merged = concatenate([self.emb_cap, self.emb_image])
    self.model = Model(inputs=[self.cap_input, self.image_input], outputs=[merged])
    self.model.compile(optimizer=tf.train.AdamOptimizer(), loss=contrastive_loss)
    
  def load_weights(self):
    # load json and create model
    with open('DLSIR_model_weights/inception+fasttext+img_self_attention/model_img_self_attention2L.json', 'r') as json_file:
      loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json, custom_objects={'Position_Embedding': Position_Embedding, 'Attention': Attention})
    # load weights into new model
    loaded_model.load_weights("DLSIR_model_weights/inception+fasttext+img_self_attention/model_weights_img_self_attention2L.h5")
    self.model = loaded_model
    print("Loaded model from disk")
    #recompile this loaded model
    self.model.compile(optimizer=tf.train.AdamOptimizer(), loss=contrastive_loss)

  def get_test_models_from_weights(self):
    weights = self.model.get_weights()
    
    im_w = weights[0:4] + weights[5:8] + weights[16:18] + weights[20:22]
    cap_w = weights[4:5] + weights[8:14] + weights[14:16] + weights[18:20]

    test_model_cap = Model(inputs=self.cap_input, outputs=self.emb_cap)
    test_model_cap.set_weights(cap_w)
    test_model_cap.compile(optimizer='adam', loss="mse")#contrastive_loss)
    test_model_im = Model(inputs=self.image_input, outputs=self.emb_image)
    test_model_im.set_weights(im_w)
    test_model_im.compile(optimizer='adam', loss="mse")#contrastive_loss)
    return test_model_cap,test_model_im

  def cache_image_embeddings(self):
    _, test_model_im = self.get_test_models_from_weights()

    img_name_test = np.load('MS_COCO/img_name_test.npy')
    
    img_name_test = list(map(lambda x: x[1:], img_name_test))
    all_test_images = img_name_test[:6000]

    all_test_image_features = list(map(lambda img_name:np.load(img_name +'.npy'), all_test_images))
    imgs_embs = test_model_im.predict(np.array(all_test_image_features))

    np.save('cached_image_embs/model_img_self_attention/imgs_embs', imgs_embs)

  def predict(self, caption, k=9):
    return get_top_k_images(self, 'cached_image_embs/model_img_self_attention/imgs_embs.npy', caption, k)

class Model_Caption_Image_Self_Attention:
  def __init__(self):
    self.image_input = Input(shape=model_config['image_input_shape'], name='image_input')
    X = Dense(2048, activation='relu')(self.image_input)
    X = Dense(1024, activation='relu')(X)
    X = Position_Embedding()(X) 
    X = Attention(model_config["num_heads"], model_config["units"])([X, X, X])
    X = GlobalAveragePooling1D()(X)
    X = Flatten()(X)
    X = Dense(2048, activation='relu')(X)
    X = Dense(model_config['output_dim'], activation='relu')(X)
    self.emb_image = Lambda(lambda x: l2norm(x))(X)
    self.cap_input = Input(shape=(model_config['max_cap_length'],), dtype='float32', name='cap_input')
    X = Masking(mask_value=0,input_shape=(model_config['max_cap_length'],))(self.cap_input)
    X = Embedding(output_dim=model_config['dim_word'], input_dim=len(tokenizer.word_index), input_length=model_config['max_cap_length'], weights=[embedding_matrix])(X)
    X = Dense(2048, activation='relu')(X)
    X = Dense(1024, activation='relu')(X)
    X = Position_Embedding()(X) 
    X = Attention(model_config["num_heads"], model_config["units"])([X, X, X])
    X = GlobalAveragePooling1D()(X)
    X = Flatten()(X)
    X = Dense(2048, activation='relu')(X)
    X = Dense(model_config['output_dim'], activation='relu')(X)
    self.emb_cap = Lambda(lambda x: l2norm(x))(X)   #the final embedding representation of the catption
    merged = concatenate([self.emb_cap, self.emb_image])
    self.model = Model(inputs=[self.cap_input, self.image_input], outputs=[merged])
    self.model.compile(optimizer=tf.train.AdamOptimizer(), loss=contrastive_loss)

  def load_weights(self):
    # load json and create model
    with open('DLSIR_model_weights/inception+fasttext+img_cap_self_attention/model_double_self_attention_2L_2.json', 'r') as json_file:
      loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json, custom_objects={'Position_Embedding': Position_Embedding, 'Attention': Attention})
    # load weights into new model
    loaded_model.load_weights("DLSIR_model_weights/inception+fasttext+img_cap_self_attention/model_weights_double_self_attention_2L_2.h5")
    self.model = loaded_model
    print("Loaded model from disk")
    #recompile this loaded model
    self.model.compile(optimizer=tf.train.AdamOptimizer(), loss=contrastive_loss) 

  def get_test_models_from_weights(self):
    weights = self.model.get_weights()
    im_w = weights[3:5] + weights[7:9] + weights[12:15] + weights[17:19] + weights[21:23]
    emb_w = weights[0:3] + weights[5:7] + weights[9:12] + weights[15:17] + weights[19:21]

    test_model_im = Model(inputs=self.image_input, outputs=self.emb_image)
    test_model_im.set_weights(im_w)
    test_model_im.compile(optimizer='adam', loss="mse")#contrastive_loss)
    test_model_cap = Model(inputs=self.cap_input, outputs=self.emb_cap)
    test_model_cap.set_weights(emb_w)
    test_model_cap.compile(optimizer='adam', loss="mse")#contrastive_loss)
    return test_model_cap,test_model_im

  def cache_image_embeddings(self):
    _, test_model_im = self.get_test_models_from_weights()

    img_name_test = np.load('MS_COCO/img_name_test.npy')
    
    img_name_test = list(map(lambda x: x[1:], img_name_test))
    all_test_images = img_name_test[:6000]

    all_test_image_features = list(map(lambda img_name:np.load(img_name +'.npy'), all_test_images))
    imgs_embs = test_model_im.predict(np.array(all_test_image_features))

    np.save('cached_image_embs/model_caption_img_self_attention/imgs_embs', imgs_embs)

  def predict(self, caption, k=9):
    return get_top_k_images(self, 'cached_image_embs/model_caption_img_self_attention/imgs_embs.npy', caption, k)
  