import streamlit as st
import pandas as pd
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

# Nur simulierte Daten erzeugen
st.markdown("---")
st.subheader("📊 Antworten anzeigen (nur Simulation)")

geschlechter = ["männlich", "weiblich", "divers (z. B. nicht-binär)", "möchte ich nicht angeben"]
lgbtq = ["ja", "nein", "lieber nicht sagen"]
pornokonsum = ["0", "1–2", "3–4", "5+"]
q1 = ["ka eig is das privat", "alle guggn halt... normal iwie", "nich jedr redet darüber aber viele tun es", "kp ich denk viele fühl sich ertappt", "is halt tabu, aba alle wissn es"]
q2 = ["kommt drauf an mit wem", "manchmal peinlich, man will nich verurteilt werden", "joa geht schon, aber nicht mit jedem", "bisschen unangenehm vllt", "nich leicht drüber zu reden"]
q3 = ["meine eltern sind streng", "in unserer Kultur redet man net darüber", "schule macht kein platz für sowas", "freunde lachen vllt drüber", "einfach scham"]

sim_data = []
for i in range(29):
    sim_data.append({
        "Geschlecht": geschlechter[i % len(geschlechter)],
        "LGBTQ+": lgbtq[i % len(lgbtq)],
        "q1": f"Antwort q1 von Person {i+1}",
        "q2": f"Antwort q2 von Person {i+1}",
        "q3": f"Antwort q3 von Person {i+1}",
        "q4": pornokonsum[i % len(pornokonsum)],
        "q5": (i % 5) + 1,
        "q6": ((i+2) % 5) + 1,
        "q7": ((i+3) % 5) + 1,
    })

data = pd.DataFrame(sim_data)
st.dataframe(data)

st.subheader("🔎 Grafische Auswertung")
fig1, ax1 = plt.subplots()
sns.countplot(data=data, x="Geschlecht", order=data["Geschlecht"].value_counts().index, ax=ax1)
ax1.set_title("Verteilung nach Geschlecht")
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
sns.countplot(data=data, x="q4", order=["0", "1–2", "3–4", "5+"], ax=ax2)
ax2.set_title("Pornokonsum pro Woche")
st.pyplot(fig2)

fig3, ax3 = plt.subplots()
sns.boxplot(data=data, x="Geschlecht", y="q5", ax=ax3)
ax3.set_title("Wie normal ist es, über Pornokonsum zu sprechen (q5)?")
st.pyplot(fig3)

# Sentimentanalyse
st.subheader("🧠 Emotionale Tendenz (Sentiment-Analyse)")
textfelder = ["q1", "q2", "q3"]
textauszüge = data[textfelder].fillna("").apply(lambda x: " ".join(x), axis=1)
sentiment_scores = textauszüge.apply(lambda x: TextBlob(x).sentiment.polarity)
st.write("Durchschnittliches Sentiment:", round(sentiment_scores.mean(), 2))
st.bar_chart(sentiment_scores)

# Themenanalyse (LDA)
st.subheader("📚 Thematische Gruppierung (LDA)")
vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=1)
dtm = vectorizer.fit_transform(textauszüge[textauszüge.str.strip() != ""])
if dtm.shape[0] >= 2 and dtm.shape[1] >= 2:
    lda = LatentDirichletAllocation(n_components=3, random_state=0)
    lda.fit(dtm)
    for idx, topic in enumerate(lda.components_):
        st.write(f"**Thema {idx+1}:**", ", ".join([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-5:][::-1]]))
