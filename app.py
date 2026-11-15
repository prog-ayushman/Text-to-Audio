import streamlit as st

from predictor import MusicPredictor


@st.cache_resource
def get_model():
    return MusicPredictor()


st.title("Text-to-Music")
with st.form("f"):
    prompt = st.text_area("Describe the music")
    seconds = st.slider("Duration (seconds)", 1, 10, 5)
    seed = st.number_input("Seed", value=0, step=1)
    go = st.form_submit_button("Generate")

if go:
    try:
        with st.spinner("Generating (this can take a few minutes on CPU)..."):
            _, rec = get_model().generate(
                prompt, seconds=seconds, seed=int(seed), output_path="outputs/last.wav"
            )
        with open("outputs/last.wav", "rb") as f:
            st.session_state["wav"] = f.read()
        st.session_state["rec"] = rec
    except ValueError as e:
        st.session_state.pop("wav", None)
        st.session_state.pop("rec", None)
        st.error(str(e))

if "wav" in st.session_state:
    st.audio(st.session_state["wav"], format="audio/wav")
    st.download_button("Download WAV", st.session_state["wav"], "clip.wav", "audio/wav")
    st.json(st.session_state["rec"])