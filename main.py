import pickle
import pandas as pd
import streamlit as st
from keras.models import load_model
import cv2
import json
from PIL import Image
import numpy as np
import urllib.request

with open('diagnosis_model.pkl', 'rb') as file:
    model = pickle.load(file)

@st.experimental_singleton
def load_my_model():
    if not os.path.isfile('model.h5'):
        urllib.request.urlretrieve('https://github.com/Elisco360/PocketDokta/blob/main/skin_diseases.h5', 'model.h5')
    return tensorflow.keras.models.load_model('model.h5')
skin_model = load_my_model()
file = open('dat.json')
skin_data = json.load(file)
skin_data_keys = list(skin_data)
data = pd.read_csv('data/Training.csv')
data = pd.DataFrame(data)
data.drop('Unnamed: 133', axis=1, inplace=True)
data.drop('prognosis', axis=1, inplace=True)


def get_skin_predictions(image):
    img = cv2.resize(image, (32,32))/float(255)
    prediction = skin_model.predict(img.reshape(1,32,32,3))
    ai_choice = skin_data_keys[prediction.argmax()]
    return ai_choice, skin_data[ai_choice]

def get_predictions(symptoms):
    reconstructed_symptoms = [1 if symptom in symptoms else 0 for symptom in data.columns]
    proba = model.predict_proba([reconstructed_symptoms])

    # Get the three highest probabilities and their corresponding labels
    top_indices = proba.argsort()[0][-3:]
    top_labels = model.classes_[top_indices]
    top_probabilities = proba[0][top_indices]
    names = []
    rate = []

    # Print the three highest labels and their probabilities
    for label, probability in zip(top_labels, top_probabilities):
        # print(f"Label: {label}, Probability: {round(probability * 100, 2)}")
        names.append(label)
        rate.append(round(probability, 2))

    return names, rate


def generate_desc(dgs):
    diseases = pd.read_csv('data/disease_description.csv')
    dds = pd.DataFrame(diseases)
    decs = []
    for d in dgs:
        decs.append(dds[dds['disease'] == d]['description'].values[0])

    return decs


st.markdown("<h1 style='text-align: center;'>Pocket Dokta 👨🏾‍⚕️</h1>", unsafe_allow_html=True)

text, image  = st.tabs(['General Diagnosis', 'Skin Disease Detection'])

with text:
    dokta = st.chat_message('assistant')

    with st.container():
        dokta.write("Hello there👋🏾, "
                    "my name is **Pocket Dokta 👨🏾‍⚕️** and I am here to assist you with preliminary diagnosis.")
        dokta.info('To optimize the nature of my feedback and accuracy of diagnosis, I have provided you a **list of '
                'symptoms you can select from** based on what you are feeling.')

        user = st.chat_message("user")
        user.write('**Symptoms**')
        all_symptoms = sorted(list(data.columns))
        user_symptoms = user.multiselect('Feel free to select as many symptoms based on how you feel',
                                    options=all_symptoms, key='stp')

        diagnose = user.button('Diagnose me')

        if diagnose and user_symptoms:
            dokta_d1 = st.chat_message('assistant')
            # dokta_d1.write('I understand that experiencing symptoms can be distressing and uncomfortable.🤗🫂')

            dokta_d1.success('The predictions provided are based on research data and may have varying levels of accuracy. '
                            'However, it is crucial to consult a doctor or visit the nearest hospital for confirmation and '
                            'appropriate treatment. Their expertise is essential in providing accurate diagnosis and '
                            'personalized care for your specific situation.')

            labels, percentages = get_predictions(user_symptoms)
            try:
                descriptions = generate_desc(labels)
            except:
                descriptions = ['','','']
            r, l, g = dokta_d1.columns(3)
            with r:
                st.metric('First possible diagnosis', f'{labels[2]}', f'{percentages[1]}')
                st.write(descriptions[2])
            with l:
                st.metric('Second possible diagnosis', f'{labels[0]}', f'{percentages[0]}')
                st.write(descriptions[0])
            with g:
                g.metric('Third Possible diagnosis', f'{labels[1]}', f'{percentages[2]}')
                st.write(descriptions[1])

with image:
    img_array = None
    check = st.selectbox('Choose you mode of input', ['','Camera Capture', 'Image Upload'])
    if check == 'Camera Capture':
        with st.expander('Click here to take picture'):
            capture = st.camera_input('Take a picture of the infected body part')
            if capture:
                img: Image = Image.open(capture)
                img_array = np.array(img)
    elif check == 'Image Upload':
        upload = st.file_uploader('Upload Image here')
        if upload:
            img: Image = Image.open(upload)
            img_array = np.array(img)
    else:
        st.warning('Please select an input mode')
    try:
        results = get_skin_predictions(image=img_array)
        if results:
            st.success('Image processed successfully ✅')
            st.markdown('<hr>', unsafe_allow_html=True)
            st.title('Results')
            st.text_input('Diagnosis', value=results[0])
            st.text_area('Description', value=results[1]['description'])
            st.text_input('Causes', value=results[1]['causes'])
            l, r = st.columns(2)
            l.text_input('Possible treatment 1', value=results[1]['treatement-1'])
            r.text_input('Possible treatment 2', value=results[1]['treatement-2'])
    except:
        pass
