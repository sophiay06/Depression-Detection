# NH
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import pandas as pd
import numpy as np

# combine all text data into a string
all_data = pd.read_parquet('../data/processed/erisk25_clean.parquet')
text_data = all_data['text'].str.cat(sep=' ')

mask_img = np.array(Image.open("image-mask.png"))
print(np.unique(mask_img))
stopwords = set(STOPWORDS)
stopwords.add("really") # add any string that feels insignificant to the word cloud, and it will be removed!
stopwords.add("thing")
stopwords.add("much")
stopwords.add("time")

wc = WordCloud(background_color="white", max_words=2000, mask=mask_img,
               stopwords=stopwords)

# generate word cloud
wc.generate(text_data)

# store to file
wc.to_file("wordcloud.png")
