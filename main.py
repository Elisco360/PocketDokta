import pickle
import pandas as pd
import streamlit as st

with open('diagnosis_model.pkl', 'rb') as file:
    model = pickle.load(file)

data = pd.read_csv('data\\Training.csv')
data = pd.DataFrame(data)
data.drop('Unnamed: 133', axis=1, inplace=True)
data.drop('prognosis', axis=1, inplace=True)


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
    diseases = pd.read_csv('data\\disease_description.csv')
    dds = pd.DataFrame(diseases)
    decs = []
    for d in dgs:
        decs.append(dds[dds['disease'] == d]['description'].values[0])

    return decs


st.markdown("<h1 style='text-align: center;'>Pocket Dokta 👨🏾‍⚕️</h1>", unsafe_allow_html=True)

dokta = st.chat_message('assistant')

with st.container():
    dokta.write("Hello there👋🏾, "
                "my name is **Pocket Dokta 👨🏾‍⚕️** and I am here to assist you with preliminary diagnosis.")
    dokta.info('To optimize the nature of our feedback and accuracy of diagnosis, I have provided you a **list of '
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
             st.write(descriptions[2])
