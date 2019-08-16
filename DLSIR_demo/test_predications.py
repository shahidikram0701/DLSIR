from models import *
from PIL import Image
import matplotlib.pyplot as plt

def show_predictions(p):
    w = 10
    h = 10
    fig=plt.figure(figsize=(25, 25))
    columns = 3
    rows = (len(p)//columns) + 1
    j = 0

    for i in range(1, columns*rows +1):
    #       img = np.random.randint(10, size=(h,w))
        fig.add_subplot(rows, columns, i)
        img = Image.open(p[j][1][1:])
        plt.imshow(img)
        
        j+=1
        
        if(j >= len(p)):
            break
        
    plt.show()

# pprint("Creating caption self attention model")
# caption_self_attention_model = Model_Caption_Self_Attention()
# pprint("Loading Weights of caption self attention model")
# caption_self_attention_model.load_weights()
# pprint("Prediction")

# caption = "women playing near the beach"

# print("caption: ", caption)
# p = caption_self_attention_model.predict(caption)
# print(p)
# show_predictions(p)


pprint("Creating Base Model")
model_base = Model_Base()
pprint("Load weights of Base Model")
model_base.load_weights()
pprint("Prediction")

caption = "men playing baseball"

print("\n\n" + caption + "\n")
p = model_base.predict(caption)
show_predictions(p)