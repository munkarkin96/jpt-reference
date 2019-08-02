#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv("11street_legit.csv")


# In[3]:


data.head()


# In[4]:


data.isnull().sum()


# In[5]:


data = data.dropna(subset=['processedData.prod_body_txt_clean', 'is_false_positive' ])


# In[6]:


data.isnull().sum()


# In[7]:


data['is_false_positive'].values


# In[8]:


list(eval(val)[0] for val in data['alert_category'].values)


# In[9]:


set(list(eval(val)[0] for val in data['alert_category'].values))


# In[10]:


img_urls = data['processedData.prod_image_links']
img_urls.head()


# In[11]:


list(map(lambda x: len(eval(x)), img_urls.values))


# eval() --> converts '[1, 2, 3]'  --> [1, 2, 3]

# In[12]:


eval(img_urls[0])


# In[13]:


test_url = 'https://www.lovefantasymart.com/image/lovefantasymart/image/data/all_product_images/product-764/set-isteri-seksi-ketat-padat-palmer-s-bust-cream-che-bunga-lovefantasymart-1708-07-lovefantasymart%4021%20%281%29.jpg'


# In[14]:


import urllib.parse

urllib.parse.unquote(test_url)


# In[15]:


test = 'https://www.11street.my/image-proxy-redirect?url=https%3A%2F%2Fimages-na.ssl-images-amazon.com%2Fimages%2FI%2F31xccEdro8L.jpg'

urllib.parse.unquote(test)


# In[16]:


search_msg = 'url='
test.find(search_msg)


# In[17]:


test[test.find(search_msg) + len(search_msg):]


# In[18]:


step_1 = urllib.parse.unquote(test)

search_msg = 'url='
step_2 = step_1[step_1.find(search_msg) + len(search_msg):]

step_2


# In[19]:


from PIL import Image
from urllib.request import urlopen
import numpy as np
import matplotlib.pyplot as plt

im = Image.open(urlopen(step_2))

np.array(im).shape


# In[20]:


plt.imshow(np.array(im))


# In[21]:


len('url=')


# In[22]:


label = list(eval(val)[0] for val in data['alert_category'].values)
urls = list(map(lambda x: eval(x), img_urls.values))
is_false_pos = data['is_false_positive'].values.tolist()

len(label), len(urls), len(is_false_pos)


# In[23]:


for i in range(len(label)):
    if is_false_pos[i]:
        label[i] = 'LEGIT'

set(label)


# In[24]:


url_sep, label_sep = [], []

for i, url in enumerate(urls):
    
    for link in url:
        url_sep.append(link)
        label_sep.append(label[i])


# In[25]:


len(url_sep), len(label_sep)


# In[26]:


d = {
    "urls": url_sep,
    "label": label_sep
}
df = pd.DataFrame(d)


# In[27]:


import multiprocessing

multiprocessing.cpu_count()


# In[28]:


df.shape


# In[ ]:


import urllib
from urllib.error import URLError, HTTPError
import urllib.request

def map_fn(data):
    link = data[1]
    label = data[0]

    track = str(data[2]) 
    

        
    step_1 = urllib.parse.unquote(link)

    if step_1.find(search_msg) != -1:
        step_2 = step_1[step_1.find(search_msg) + len(search_msg):]
    else:
        step_2 = step_1 

    try:
        im = Image.open(urlopen(step_2))
    except HTTPError as e:
        # print(e)
        print(track, 'The server couldn\'t fulfill the request, ', e)
        print(track, link, step_2)

        if e.reason[0] == 403 or e.reason[0] == 400:
            hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
                   'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                   'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
                   'Accept-Encoding': 'none',
                   'Accept-Language': 'en-US,en;q=0.8',
                   'Connection': 'keep-alive'}

            step_2 = urllib.request.Request(site, headers=hdr)
        else:
            return None
        
        # HTTP Error 400: Bad Request (didn't handle this error, but not a lot anyway (seen: 1))


    except URLError as e:
        print(track, 'Failed to reach a server, ', e)
        print(track, link, step_2)
        return None

    except ValueError as e:
        step_2 = step_2.replace("//", "http://")
        #im = Image.open(urlopen(step_2))


    except Exception as e:
        print(track, 'Wild card, ', e)
        print(track, link, step_2)
        return None
    # i.e. 
    #  <urlopen error [Errno -3] Temporary failure in name resolution>

    try:
        im = Image.open(urlopen(step_2))
    except:
        return None

    path = os.getcwd() + '/' + 'OSM_img/' + label 

    if not os.path.exists(path):
        os.mkdir(path)
    else:
        pass

    path = 'OSM_img/' + label + "/" + track + ".png" #re.sub(r'\W+', '', link) + ".png"
    im.save(path ,"PNG")
    #im_np = np.array(im)
    
    return None
    
    
#im = []

from multiprocessing import Pool
import os
import re

df_target = df[df['label'] == 'LEGIT']

print(len(df_target))

p = Pool(multiprocessing.cpu_count())
im = p.map(map_fn,  list(zip(df_target['label'].values, df_target['urls'], range(len(df_target)))))


# In[ ]:


#len(im), type(im), len(df)


# In[ ]:


import pickle 
with open('OSM_image_data_np.pickle', 'wb') as f:
    pickle.dump(im, f)

"""
with open('OSM_image_data_np.pickle', 'rb') as f:
    im = pickle.load(f)
"""


# In[ ]:





# ![image.png](attachment:image.png)
# 
# https://docs.python.org/3.1/howto/urllib2.html

# In[ ]:


import urllib
from urllib.error import URLError, HTTPError
import urllib.request


X, y = [], []
search_msg = 'url='

m = len(urls)

for i, url in enumerate(urls):
    # step 1: replace UTF-8 characters
    # step 2: remove everything before url=

    for link in url: 
        step_1 = urllib.parse.unquote(link)

        if step_1.find(search_msg) != -1:
            step_2 = step_1[step_1.find(search_msg) + len(search_msg):]
        else:
            step_2 = step_1 

        try:
            im = Image.open(urlopen(step_2))
        except HTTPError as e:
            # print(e)
            print(i, 'The server couldn\'t fulfill the request, ', e)
            print(link, step_2)

            if e.reason[0] == 403:
                hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
                       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
                       'Accept-Encoding': 'none',
                       'Accept-Language': 'en-US,en;q=0.8',
                       'Connection': 'keep-alive'}

                step_2 = urllib.request.Request(site, headers=hdr)

                #im = Image.open(urlopen(step_2))

            else:
                continue 

        except URLError as e:
            print(i, 'Failed to reach a server, ', e)
            print(link, step_2)
            continue 

        except ValueError as e:
            step_2 = step_2.replace("//", "http://")
            #im = Image.open(urlopen(step_2))


        except Exception as e:
            print(i, 'Wild card, ', e)
            print(link, step_2)
            continue 

        try:
            im = Image.open(urlopen(step_2))
        except:
            continue 
        
        im_np = np.array(im)
        
        X.append(im_np)
        y.append(label[i])


# In[ ]:


len(urls)


# In[ ]:


len(X), len(y)


# ## Error Exception Documentations

# In[ ]:


test_l = "https://www.lovefantasymart.com/image/lovefantasymart/image/data/all_product_images/product-764/oral-sex-vibrater-rotating-tongue-head-lovefantasymart-1802-27-lovefantasymart@2 (1).jpg"


# In[ ]:


urlopen(test_l)


# In[ ]:


site = "https://www.lovefantasymart.com/image/lovefantasymart/image/data/all_product_images/product-764/oral-sex-vibrater-rotating-tongue-head-lovefantasymart-1802-27-lovefantasymart@2 (1).jpg"
hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}

import urllib.request
req = urllib.request.Request(site, headers=hdr)

Image.open(urlopen(req))


# ## Data Exploration

# In[ ]:





# In[ ]:





# In[ ]:





# ## Training
# 
# Note: Let's try to use ImageDataGenerator this time 
# 
# https://medium.com/@ksusorokina/image-classification-with-convolutional-neural-networks-496815db12a8

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

