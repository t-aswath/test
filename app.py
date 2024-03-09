import gradio as gr
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.models import load_model

model = load_model('cimta.h5')

def greet(num_patients,avg_wait_time,distance_to_hospital,doc_rating):
        y_predict = model.predict([[num_patients,avg_wait_time,distance_to_hospital,doc_rating]])
        y_predict = (y_predict.reshape(-1,))
        return y_predict[0]


iface = gr.Interface(fn=greet, inputs=["number","number","number","number"], outputs="text")
iface.launch(share=True)