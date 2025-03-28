import streamlit as st
import pandas as pd
import os

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

# Bereich für Admin / Forscher: Daten anzeigen
st.markdown("---")
with st.expander("📊 Antworten anzeigen (Admin)"):
    try:
        data = pd.read_csv("antworten.csv")
        st.dataframe(data)
    except FileNotFoundError:
        st.warning("Noch keine Antworten gespeichert oder Datei nicht gefunden.")
    except Exception as e:
        st.error(f"Fehler beim Laden der Daten: {e}")
