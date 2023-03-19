#importing libs
import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open(r'C:\Users\agnis\Desktop\Workspace\DEV\MACHINE LEARNING PROJECTS\DISEASE DETECTION MODELS\Heart Disease Prediction\trained_model.sav','rb'))


# creating a function for model
def heart_risk_prediction(input_data):
    input_data_as_numpy_array=np.asarray(input_data)
    input_data_reshaped=input_data_as_numpy_array.astype(float).reshape(1,-1)
    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)
    if(prediction==1):
       return "Bad Heart Condition"
    else:
       return "Good Health Condition"




def main():

    #title for user interface
    st.title('HEART RISK PREDICTION WEB APP')


    #getting the input data from user
    age =st.text_input('Age in years')	
    sex	=st.text_input('sex (1 = male; 0 = female)')	
    cp	=st.text_input('chest pain type ')	
    restbps=st.text_input(' resting blood pressure (in mm Hg on admission to the hospital) ')		
    chol =st.text_input('serum cholestoral in mg/dl')	
    fbs	=st.text_input('(fasting blood sugar &gt; 120 mg/dl) (1 = true; 0 = false)')	
    restecg	=st.text_input('resting electrocardiographic results')	
    thalach	=st.text_input('maximum heart rate achieved')	
    exang =st.text_input('exercise induced angina (1 = yes; 0 = no)')	
    oldpeak	=st.text_input('ST depression induced by exercise relative to rest')	
    slope=st.text_input('the slope of the peak exercise ST segment')		
    ca =st.text_input('number of major vessels (0-3) colored by flourosopy')	
    thal =st.text_input(' 0 = normal; 1 = fixed defect; 2 = reversable defect')	
   

   #code for prediction

    diagnosis=''

   #creating a button for prediction

    if st.button('Heart Test Result'):
       diagnosis=heart_risk_prediction([age,sex,cp,restbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])

    st.success(diagnosis)



if __name__=='__main__':
    main()

