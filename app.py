import streamlit as st
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

st.set_page_config(page_title="Studie zur emotionalen Offenheit", layout="centered")
st.title("Online-Studie: Emotionale Ausdrucksfähigkeit und Diversität")

st.markdown("""
Diese Studie untersucht, wie **Geschlecht** und **Diversität** die Offenheit von Jugendlichen beim Thema **Pornokonsum** beeinflussen. 
Die Antworten sind anonym und dienen ausschließlich wissenschaftlichen Zwecken.
""")

# Auswahl des Modus
modus = st.radio("Bitte wähle den Fragetyp:", ["Qualitativ", "Quantitativ", "Kombiniert"])

antworten = {}

# Geschlecht und LGBTQ+-Zugehörigkeit
antworten["Geschlecht"] = st.selectbox("Was ist dein Geschlecht?", ["männlich", "weiblich", "divers (z. B. nicht-binär)", "möchte ich nicht angeben"])
antworten["LGBTQ+"] = st.radio("Identifizierst du dich als Teil der LGBTQ+ Community?", ["ja", "nein", "lieber nicht sagen"])

# Fragen je nach Modus
st.markdown("---")

if modus == "Qualitativ":
    antworten["q1"] = st.text_area("Was denkst du über den Pornokonsum unter Jugendlichen?")
    antworten["q2"] = st.text_area("Fällt es dir leicht oder schwer, über eigene Erfahrungen zu sprechen? Warum?")
    antworten["q3"] = st.text_area("Gibt es gesellschaftliche oder familiäre Gründe, warum du dich zu diesem Thema nicht äußerst?")

elif modus == "Quantitativ":
    antworten["q4"] = st.selectbox("Wie oft konsumierst du pornografische Inhalte pro Woche?", ["0", "1–2", "3–4", "5+"])
    antworten["q5"] = st.slider("Ich finde es normal, über Pornokonsum zu sprechen.", 1, 5, 3)
    antworten["q6"] = st.slider("Ich fühle mich wohl, wenn ich über mein eigenes Verhalten spreche.", 1, 5, 3)
    antworten["q7"] = st.slider("Ich fühle gesellschaftlichen Druck, mich dazu nicht zu äußern.", 1, 5, 3)

elif modus == "Kombiniert":
    antworten["q4"] = st.selectbox("Wie oft konsumierst du pornografische Inhalte pro Woche?", ["0", "1–2", "3–4", "5+"])
    antworten["q5"] = st.slider("Ich finde es normal, über Pornokonsum zu sprechen.", 1, 5, 3)
    antworten["q5_erklärung"] = st.text_input("Möchtest du kurz erklären, warum?")
    antworten["q6"] = st.slider("Ich fühle mich wohl, wenn ich über mein eigenes Verhalten spreche.", 1, 5, 3)
    antworten["q6_erklärung"] = st.text_input("Was beeinflusst dieses Gefühl?")
    antworten["q7"] = st.slider("Ich fühle gesellschaftlichen Druck, mich dazu nicht zu äußern.", 1, 5, 3)
    antworten["q7_erklärung"] = st.text_input("Hast du konkrete Beispiele dafür?")

# Absenden
st.markdown("---")
if st.button("Antworten speichern"):
    df = pd.DataFrame([antworten])
    if os.path.exists("antworten.csv"):
        df.to_csv("antworten.csv", mode='a', header=False, index=False)
    else:
        df.to_csv("antworten.csv", index=False)
    st.success("Deine Antworten wurden gespeichert. Vielen Dank für deine Teilnahme!")

# Bereich für Admin / Forscher: Daten anzeigen und analysieren
st.markdown("---")
with st.expander("📊 Antworten anzeigen (Admin)"):
    try:
        if os.path.exists("antworten.csv"):
            data = pd.read_csv("antworten.csv", on_bad_lines='skip')
        else:
            data = pd.DataFrame()

        st.dataframe(data)

        st.subheader("🔎 Grafische Auswertung")
        if not data.empty:
            if "Geschlecht" in data.columns:
                fig1, ax1 = plt.subplots()
                sns.countplot(data=data, x="Geschlecht", order=data["Geschlecht"].value_counts().index, ax=ax1)
                ax1.set_title("Verteilung nach Geschlecht")
                st.pyplot(fig1)

            if "q4" in data.columns:
                fig2, ax2 = plt.subplots()
                sns.countplot(data=data, x="q4", order=["0", "1–2", "3–4", "5+"], ax=ax2)
                ax2.set_title("Pornokonsum pro Woche")
                st.pyplot(fig2)

            if "q5" in data.columns and "Geschlecht" in data.columns:
                fig3, ax3 = plt.subplots()
                sns.boxplot(data=data, x="Geschlecht", y="q5", ax=ax3)
                ax3.set_title("Wie normal ist es, über Pornokonsum zu sprechen (q5)?")
                st.pyplot(fig3)

        # Sentimentanalyse
        st.subheader("🧠 Emotionale Tendenz (Sentiment-Analyse)")
        textfelder = ["q1", "q2", "q3", "q5_erklärung", "q6_erklärung", "q7_erklärung"]
        textauszüge = data[textfelder].fillna("").apply(lambda x: " ".join(x), axis=1)
        sentiment_scores = textauszüge.apply(lambda x: TextBlob(x).sentiment.polarity)
        st.write("Durchschnittliches Sentiment:", round(sentiment_scores.mean(), 2))
        st.bar_chart(sentiment_scores)

        # Themenanalyse (LDA)
        st.subheader("📚 Thematische Gruppierung (LDA)")
        vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=2)
        dtm = vectorizer.fit_transform(textauszüge)
        if dtm.shape[0] >= 2 and dtm.shape[1] >= 2:
            lda = LatentDirichletAllocation(n_components=3, random_state=0)
            lda.fit(dtm)
            for idx, topic in enumerate(lda.components_):
                st.write(f"**Thema {idx+1}:**", ", ".join([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-5:][::-1]]))

    except Exception as e:
        st.error(f"Fehler beim Laden der Daten: {e}")
