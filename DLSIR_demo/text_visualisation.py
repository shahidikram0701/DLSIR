# Python program to generate WordCloud 

# importing all necessery modules 
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 
import json

# Reads 'Youtube04-Eminem.csv' file 
#df = pd.read_csv(r"Youtube04-Eminem.csv", encoding ="latin-1") 
annotation_file = 'annotations/captions_train2014.json'
# read the json file
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# storing the captions and the image name in vectors
all_captions = []# only has the captions
all_img_name_vector = []# only has the paths

for annot in annotations['annotations']:
    #have to see if each image has 1:5 relation with the captions and have to incorporate that somehow!
    caption = '<start> ' + annot['caption'] + ' <end>'
    #image_id = annot['image_id']
    #full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)
    
    #all_img_name_vector.append(full_coco_image_path)
    all_captions.append(caption)



comment_words = ' '
stopwords = set(STOPWORDS) 

# iterate through the csv file 
for val in all_captions[:6000]: 
	
	# typecaste each val to string 
	val = str(val) 

	# split the value 
	tokens = val.split() 
	
	# Converts each token into lowercase 
	for i in range(len(tokens)): 
		tokens[i] = tokens[i].lower() 
		
	for words in tokens: 
	    comment_words = comment_words + words + ' '


wordcloud = WordCloud(width = 800, height = 800, 
				background_color ='white', 
				stopwords = stopwords, 
				min_font_size = 10).generate(comment_words) 

# plot the WordCloud image					 
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 

plt.show() 
