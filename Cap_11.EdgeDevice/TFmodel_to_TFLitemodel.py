"""
Created on July 29 14:43:20 2021
Descripcion: Script para convertir de un modelo hecho en tensorflow a uno ajustado a tensorflow
lite.

"""
#%%
import tensorflow as tf
#%% Metodos
# Save Model
export_dir = "Pruea solamente "
model = "Modelo de Keras"
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir) # El argumento es el directorio del modelo guardado .pb

# Keras Model
converter = tf.lite.TFLiteConverter.from_keras_model_file(model) # El argumento es un modelo de la api de keras

# Concrete Functions
converter = tf.lite.TFLiteConverter.from_concrete_functions(["concrete_func"]) # El argumento es una funcion del desarrollador
#%% Ejemplo de como serializar un modelo

# Usando Save model
tf.saved_model.save("<objeto>",'Directorio')
tf.saved_model.save('obj','export_dir',signatures=None,options=None)

# Usando Keras model
tf.keras.model.save('<objeto.h5>') # Despues de compilado un modelo este arquiere el metodo save de la api
#%% Optimizando el tamaño del modelo

converter = tf.lite.TFLiteConverter.from_saved_model('save_model_dir')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
#%% Cuantizacion de enteros

converter = tf.lite.TFLiteConverter.from_saved_model('save_model_dir')
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8