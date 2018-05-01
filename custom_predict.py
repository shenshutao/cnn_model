import os
import re

import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing import image

def preprocess_input(x):
    x /= 255.    # normalization
    x -= 0.5
    x *= 2.
    return x


def do_predict(model_h5, weights_h5, categories, img_width, img_height, input_folder, output_file):
    image.LOAD_TRUNCATED_IMAGES = True

    model = load_model(model_h5)
    if weights_h5 is not None:
        model.load_weights(weights_h5)

    rows = []
    column_names = ['id', 'category']
    for f in os.listdir(input_folder):
        if not f.startswith('.'):
            try:
                img = image.load_img(input_folder + '/' + f, target_size=(img_width, img_height))
                img_array = image.img_to_array(img)
                x = np.expand_dims(img_array, axis=0)
                x = preprocess_input(x)
                y_prob = model.predict(x)
                y_classes = y_prob.argmax(axis=-1)
                cat_id = y_classes[0]

                row = [str(f), str(categories[cat_id])]
                rows.append(row)
            except Exception as e:
                print e.message
                print 'Canot predict image: ' + f

    df = pd.DataFrame(rows, columns=column_names)
    df.to_csv(output_file, index=False, header=True)
    print 'Done'

