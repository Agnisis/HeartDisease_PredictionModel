import numpy as np
import pickle

loaded_model = pickle.load(open('trained_model.sav', 'rb'), encoding='latin1')

input_data=(41,0,1,130,204,0,0,172,0,1.4,2,0,2)
##numpy array conversion from tupple
input_data_as_numpy_array=np.asarray(input_data)

##reshaping the array to tell model to predicition only one instance

input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=loaded_model.predict(input_data_reshaped)
print(prediction)

if(prediction==1):
  print("bad heart condition")
else:
  print("good health condition")
