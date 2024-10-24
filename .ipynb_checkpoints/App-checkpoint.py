import gradio as gr
import tensorflow as tf
from fastai.learner import load_learner
import numpy as np
from PIL import Image

modelo = load_learner('model.pkl')

def clasificar_imagen(imagen):
    imagen = imagen.resize((224, 224))
    imagen_np = np.array(imagen)
    imagen_np = np.expand_dims(imagen_np, axis=0)
    imagen_np = preprocess_input(imagen_np)
    
    predicciones = modelo.predict(imagen_np)
    etiquetas = decode_predictions(predicciones, top=3)[0]
    
    resultados = {etiqueta[1]: float(etiqueta[2]) for etiqueta in etiquetas}
    return resultados

interfaz = gr.Interface(fn=clasificar_imagen, inputs="image", outputs="label")

interfaz.launch()
