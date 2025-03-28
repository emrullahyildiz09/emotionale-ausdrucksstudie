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
individuelle_q1 = [
    "Ich denke, das Thema ist sehr sensibel und wird oft totgeschwiegen.",
    "Viele meiner Freunde schauen regelmäßig Pornos, es ist fast normal geworden.",
    "Ich finde, das ist eine private Angelegenheit, über die man nicht öffentlich redet.",
    "Die Gesellschaft tabuisiert das Thema zu sehr.",
    "Pornokonsum ist nicht per se schlecht, aber es kommt auf die Häufigkeit und Inhalte an.",
    "Ich glaube, Jugendliche reden selten offen darüber, weil es ihnen peinlich ist.",
    "Das Thema ist in unserer Familie ein No-Go.",
    "Ich habe gemischte Gefühle – einerseits Neugier, andererseits Scham.",
    "Es wird viel konsumiert, aber kaum reflektiert.",
    "Das Internet macht den Zugang zu leicht.",
    "Ich finde es wichtig, dass man auch in der Schule darüber spricht.",
    "Jede*r soll das für sich entscheiden, aber ehrlich sein wäre schön.",
    "Viele reden anonym online darüber, aber nicht im echten Leben.",
    "Ich sehe es als Teil der sexuellen Entwicklung.",
    "Es gibt kaum sichere Räume, wo man darüber sprechen kann.",
    "Man sollte die positiven und negativen Seiten diskutieren dürfen.",
    "Das Thema ist mit vielen Vorurteilen behaftet.",
    "Ich habe damit früh Erfahrungen gemacht, das prägt.",
    "Pornos zeigen oft unrealistische Vorstellungen.",
    "Es ist schwierig, die Grenze zwischen gesundem Konsum und Sucht zu ziehen.",
    "Ich kenne niemanden, der ganz offen darüber spricht.",
    "Oft wird man als komisch abgestempelt, wenn man es anspricht.",
    "Ich glaube, es ist normal – aber niemand will es zugeben.",
    "Die Medien beeinflussen stark, was als akzeptabel gilt.",
    "Das ist ein Thema, das viele beschäftigt, aber keiner traut sich.",
    "Ich habe nie mit Erwachsenen darüber gesprochen.",
    "Man sollte respektvoll darüber reden können.",
    "Jeder Mensch hat andere Erfahrungen damit.",
    "Ich finde, es gehört zur Realität von Jugendlichen."
]

for i in range(29):
    individuelle_q2 = [
        "Es fällt mir schwer, weil ich Angst vor Verurteilung habe.",
        "Ich rede lieber anonym als mit Bekannten.",
        "Nur mit engen Freunden kann ich offen sprechen.",
        "Ich bin sehr verschlossen bei persönlichen Themen.",
        "Kommt drauf an, wer fragt.",
        "Wenn ich Vertrauen habe, rede ich ehrlich.",
        "Ich brauche Zeit, um mich zu öffnen.",
        "Ich habe gute Erfahrungen gemacht, wenn ich ehrlich war.",
        "Ich hatte schon negative Reaktionen.",
        "Ich bin offen, aber nicht bei allem.",
        "Ich versuche, ehrlich zu sein, auch wenn es schwer ist.",
        "Ich schäme mich manchmal für meine Erfahrungen.",
        "Ich finde es mutig, darüber zu reden.",
        "Ich will nicht als komisch abgestempelt werden.",
        "Ich habe Angst, verletzt zu werden.",
        "Ich finde Worte oft nicht für das, was ich erlebt habe.",
        "Es ist unangenehm, aber manchmal notwendig.",
        "Ich habe nie gelernt, über Gefühle zu sprechen.",
        "Ich rede lieber schriftlich als mündlich.",
        "Ich kann das nur mit bestimmten Menschen.",
        "Ich brauche ein sicheres Umfeld.",
        "Es hängt von meiner Stimmung ab.",
        "Ich habe das früher vermieden, jetzt übe ich es.",
        "Ich denke, es hilft, aber es kostet Überwindung.",
        "Ich bin offen, solange ich ernst genommen werde.",
        "Ich wurde früher ausgelacht, das hat mich geprägt.",
        "Ich fühle mich oft unverstanden.",
        "Es tut gut, wenn man wirklich zuhört.",
        "Ich hätte gerne mehr Unterstützung dabei."
    ]
    individuelle_q3 = [
        "In meiner Kultur wird über Sexualität kaum gesprochen.",
        "Meine Eltern vermeiden solche Themen komplett.",
        "Ich habe Angst, dass ich Ärger bekomme.",
        "Religion spielt bei uns eine große Rolle.",
        "Ich fühle mich unwohl, wenn solche Themen auftauchen.",
        "Bei uns zuhause wird das als Tabu gesehen.",
        "Ich darf so etwas nicht einmal denken.",
        "Ich wüsste nicht, mit wem ich reden soll.",
        "In der Schule fehlt der Raum dafür.",
        "Ich wurde früher für solche Fragen bestraft.",
        "Es gibt niemanden, der offen ist.",
        "Ich habe Angst vor dem Urteil anderer.",
        "Ich will nicht auffallen.",
        "Es ist peinlich, darüber zu reden.",
        "Ich denke, das gehört sich nicht.",
        "Ich wurde so erzogen, dass man schweigt.",
        "Ich glaube, es wäre falsch, sich zu äußern.",
        "Ich fühle mich nicht frei, sowas zu teilen.",
        "Es gibt viele Vorurteile.",
        "Ich habe das nie gelernt.",
        "Ich weiß nicht, wie ich anfangen soll.",
        "Ich habe schlechte Erfahrungen gemacht.",
        "Es ist mir einfach unangenehm.",
        "Ich denke, es ist zu privat.",
        "Ich werde innerlich blockiert.",
        "Ich traue mich nicht.",
        "Ich fühle mich dadurch angreifbar.",
        "Ich befürchte Ablehnung.",
        "Ich habe Angst, missverstanden zu werden."
    ]
    sim_data.append({
        "Geschlecht": geschlechter[i % len(geschlechter)],
        "LGBTQ+": lgbtq[i % len(lgbtq)],
        "q1": individuelle_q1[i],
        "q2": individuelle_q2[i],
        "q3": individuelle_q3[i],
        "q4": pornokonsum[i % len(pornokonsum)],
        "q5": (i % 5) + 1,
        "q6": ((i+2) % 5) + 1,
        "q7": ((i+3) % 5) + 1,
    })

data = pd.DataFrame(sim_data)
st.write("### 🧾 Einzelne Antworten")
for i, row in data.iterrows():
    st.markdown(f"**Person {i+1}**")
    st.markdown(f"• Geschlecht: {row['Geschlecht']} | LGBTQ+: {row['LGBTQ+']} | q4: {row['q4']}")
    st.markdown(f"• q1: _{row['q1']}_")
    st.markdown(f"• q2: _{row['q2']}_")
    st.markdown(f"• q3: _{row['q3']}_")
    st.markdown("---")

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

fig3, ax3 = plt.subplots(figsize=(10, 5))
sns.boxplot(data=data, x="Geschlecht", y="q5", ax=ax3)
ax3.tick_params(axis='x', labelrotation=15)
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

