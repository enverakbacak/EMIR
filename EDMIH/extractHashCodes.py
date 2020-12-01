#coding=utf-8
from keras.models import Sequential,Input,Model,InputLayer
from keras.models import model_from_json
from keras.models import load_model
import numpy as np    # for mathematical operations

np.set_printoptions(linewidth=1024)


json_file = open('./models/edmih_128_2_model.json', 'r')
model = json_file.read()
json_file.close()
model = model_from_json(model)
model.load_weights("./models/edmih_128_2_weights.h5")
model = Model(inputs=model.input, outputs=model.get_layer('dense_2').output)

X = np.load(open('preprocessed_X2.npy'))
X.shape

features = model.predict(X, batch_size=64, verbose=0, steps=None) #     TRY THIS ALSO XX = model.predict(X, batch_size)
features = features.astype(float)
np.savetxt('hashCodes/features_2_128_2.txt',features, fmt='%f')
features = np.sign(features)
features = features.astype(int)
np.savetxt('hashCodes/hashCodes_128_2.txt',features, fmt='%d')


