# FASE DE IMPORTAR DEPENDENCIAS:
import numpy as np
import math
import re
import time
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
print('Importación de dependencias')

# FASE DE CARGA DE DATOS:
print('Carga de datos')

with open("data/europarl-v7.es-en.en",
          mode="r", encoding="utf-8") as f:
    europarl_en = f.read()
with open("data/europarl-v7.es-en.es",
          mode="r", encoding="utf-8") as f:
    europarl_es = f.read()
with open("data/nonbreaking_prefix.en",
          mode="r", encoding="utf-8") as f:
    non_breaking_prefix_en = f.read()
with open("data/nonbreaking_prefix.es",
          mode="r", encoding="utf-8") as f:
    non_breaking_prefix_es = f.read()

print(europarl_en[:100])
print(europarl_es[:100])