# DLSIR

In the modern era, an enormous amount of digital pictures, from personal photos to medical images, is produced and stored every day. It is more and more common to have thousands of photos sitting in our smart phones; however, what comes with the convenience of recording unforgettable moments is the pain of searching for a specific picture or frame. How nice it would be to be able to find the desired image just by typing one or few words to describe it? In this context, automated caption image retrieval is becoming an increasingly
attracting feature, comparable to text search.

In this project, we consider the task of content-based image retrieval and propose effective neural network-based solutions for that. Specifically, the input to our algorithm is a collection of raw images in which the user would like to search, and a query sentence meant to describe the desired image. The output of the algorithm would be a list of top images that we think are relevant to the query sentence.

In particular, we obtain a representation of the sentence that will be properly align with the corresponding image features in a shared high dimensional space. The images are found based on nearest neighborhood search in that shared space.

## Focus of the Project

- Better Captioning using attention
    - The focus will be on incorporating attention into the baseline model to see if there will be any improvements.
    - Usage of various word embedding models
    - The solution involves experimenting with various word embeddings and evaluating them for the caption dataset.

- Usage of various CNN models
    - The solution involves experimenting with multiple convolutional neural network models via transfer learning and evaluating which model gives out the best results in the long run.

- Building on top of a state of the art base model
    - The base models being used for this model are best in class and hence provide the best representation for both the image and the captions.
    - E.g. In most of our models, the images are represented with the feature vectors derived from Google’s Inception CNN model whereas the captions are represented with the feature vectors derived from Facebook’s Fasttext word embedding model.
    - Both of the above models have been used time and again to achieve a state of the performance in the field of computer vision and natural language processing respectively

## Experiments conducted

As a part of this project we have tried out a multitude of experiments where each new model developed had some minor architectural differences which gave it a slight edge over its previous model but the overall arching encoder-decoder architecture still remains the same.

Listed below are all the experiments we conducted so far:
  * Baseline models
    * Resnet + GRU + fasttext
    * Inception + GRU + fastext
  * Baseline with bidirectional
    * Inception + Bidirectional(LSTM) + fastext
  * Self attention models
    * SelfAttention(Inception) + Bidirectional(LSTM) + fastext
    * Inception + SelfAttention(fastext)
    * SelfAttention(inception) + SelfAttention(fastext)
    
## Results

![](/metrics/results.JPG)

In the models column, each model has been named in the following format:

> MODEL_NAME [configuration]
`configuration refers to layers used in the model architecture.`

Here the models labelled as SA1, SA2 and DSA refer to models which are different experiments conducted by using the “Self Attention” layer found in “transformer networks”.
  1. SA1: Self Attention Model 1 where we used the multi head self-attention layer as a part of the decoder side. i.e on the images.
  2. SA2: Self Attention Model 2 where we used the multi head self-attention layer as a part of the encode side. i.e on the captions
  3. DSA: Double Self Attention Model where we used the multihead self-attention layer in both the encoder and the decoder side of the model architecture.

## How to run the project

To start with,
download the weights of the pretrained models from [here](https://mega.nz/#F!zow0XCRT!xlSu9UGgAKO56gszuTQkdQ),  and save the folder in the same directory as the other files of this repo.

Now the base directory will have 3 subdirs which are namely:
  1.	DLSIR_demo: contains the code for the final GUI demo
  2.	DLSIR_model_training_experiments: contains several colab notebooks where different experiments where trained and the model weights were saved.
  3.	DLSIR_model_weights: contains several model weights for the various models which were trained.

In order to run the DLSIR_demo the steps to follow are:
  1.	Run download_data.py
  2.	Run inception_features_saving_to_disk.py
  3.	Run cache_image_embeddings.py
  4.	Run predictions.py and server.py as two separate simultaneous processes
  5.	Go to 0.0.0.0:5000 to see the application running
  
### Developers:
  - [`Parashara Ramesh`](https://github.com/ParasharaRamesh)
  - [`Shahid Ikram`](https://github.com/shahidikram0701)
  - [`Sumanth Rao`](https://github.com/sumanthrao)
