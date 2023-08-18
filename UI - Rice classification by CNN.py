#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[1]:


import streamlit as st 
from PIL import Image

import torchvision.transforms as transforms
import torch
import torchvision.models as models
import torchvision


# # Transformer for input image

# In[2]:


transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((227,227)),
                                transforms.Normalize((0.4914, 0.4822, 0.4465) , (0.2023, 0.1994, 0.2010))])


# # Import the model

# In[3]:


AlexNet = models.alexnet(pretrained=False)

num_features = AlexNet.classifier[6].in_features
AlexNet.classifier[6] = torch.nn.Linear(num_features,5)


# In[4]:


StateDict = torch.load(r"D:\IT\IMT Courses\Deep Learning\CNN\Projects\P2_Final_Project_pytorch_Transfer Learning.pth")
AlexNet.load_state_dict(StateDict)


# # Define predictor function

# In[5]:


AlexNet.eval()
def predictor(image , model):
    image = transform(image) 
    image = image.unsqueeze(0)
    predict = model(image)
    predicted_class_index = torch.argmax(predict).item()
    return predicted_class_index


# In[6]:


st.header('Rice types image classification')
st.success('We predict types of your rice by CNN, for using please enter your rice image')

upload_file = st.file_uploader('Please enter your image rice' , type=["jpg", "jpeg", "png"])

if upload_file is not None :
    image = Image.open(upload_file)
    st.image(image , caption='Uploaded image')
    
    if st.button('Classify'):
        predicted_class_index = predictor(image , AlexNet)
        classes = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
        predict_class_label = classes[predicted_class_index]
        st.success('Predicted type is : ' + predict_class_label)


# In[ ]:





# In[ ]:




