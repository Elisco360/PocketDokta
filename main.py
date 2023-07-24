import pickle
import pandas as pd
import streamlit as st
import cv2
import json
from PIL import Image
import numpy as np
import requests
import plotly.express as px
import time


links = {
    "Melanoma": "https://www.wikipedia.org/wiki/Melanoma",
    "Vascular-lesions": "https://en.wikipedia.org/wiki/Vascular_anomaly",
    "Benign-keratosis-like-lesions":"https://www.mayoclinic.org/diseases-conditions/seborrheic-keratosis/symptoms-causes/syc-20353878#:~:text=A%20seborrheic%20keratosis%20(seb%2Do,or%20scaly%20and%20slightly%20raised.",
    "Basal-cell-carcinoma":"https://www.mayoclinic.org/diseases-conditions/basal-cell-carcinoma/symptoms-causes/syc-20354187#:~:text=on%20brown%20skin-,Basal%20cell%20carcinoma%20is%20a%20type%20of%20skin%20cancer%20that,a%20type%20of%20skin%20cancer.",
    "Melanocytic-nevi":"https://en.wikipedia.org/wiki/Melanocytic_nevus",
    "Actinic-keratoses":"https://www.mayoclinic.org/diseases-conditions/actinic-keratosis/symptoms-causes/syc-20354969",
    "Dermatofibroma":"https://dermnetnz.org/topics/dermatofibroma",
}


API_URL = "https://api-inference.huggingface.co/models/gianlab/swin-tiny-patch4-window7-224-finetuned-skin-cancer"
API_TOKEN = "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"
headers = {"Authorization": f"Bearer {API_TOKEN}"}


st.set_page_config(page_title='Pocket Dokta', page_icon='👨‍⚕️')

with open('diagnosis_model.pkl', 'rb') as file:
    model = pickle.load(file)


data = pd.read_csv('data/Training.csv')
data = pd.DataFrame(data)
data.drop('Unnamed: 133', axis=1, inplace=True)
data.drop('prognosis', axis=1, inplace=True)


def query(data):
    """Send a request to the API and return the response."""
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

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
                img = Image.open(capture)
                img_array = capture
    elif check == 'Image Upload':
        upload = st.file_uploader('Upload Image here')
        if upload:
            img: Image = Image.open(upload)
            img_array = upload
    else:
        st.warning('Please select an input mode')
        
    if img_array is not None:
        image_bytes = img_array.getvalue()
        st.subheader("Model output")

        with st.spinner("Waiting for the prediction..."):
            data = query(image_bytes)
            while "error" in data:
                time.sleep(4)
                data = query(image_bytes)
        st.success("Done!")

        scores = [e['score'] for e in data]

        labels = []
        for e in data:
            if e['label'] in links:
                labels.append(f"<a href='{links[e['label']]}' target='_blank'>{e['label']}</a>")
            else:
                labels.append(e['label'])

        df = pd.DataFrame({'scores': scores, 'labels': labels})
        #st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)
        fig = px.bar(df, x='scores', y='labels', orientation='h', color='scores')
        st.write(fig)
        st.info("""
            The scores above represent the probability of having a skin cancer
            of a particular type. The higher the score, the higher the probability.
            Note that the model is not perfect, and one should see a doctor for
            a professional observation.

            **Also, if the model returns a score of below 0.3 for all the classes, it means
            that the lesion is probably not a skin cancer.**
        """)
        





    
