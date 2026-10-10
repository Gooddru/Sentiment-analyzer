import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Анализатор тональности", page_icon="🎭")
st.title("🎭 Анализатор эмоций в тексте")


@st.cache_resource
def load_model():
    return pipeline("text-classification", model="blanchefort/rubert-base-cased-sentiment")


model = load_model()

user_text = st.text_area("Введите текст для анализа:", height=150,
                         placeholder="Например: Мне очень понравился этот фильм, актеры просто супер!")

if st.button("Анализировать текст 🚀"):
    if user_text.strip() == "":
        st.warning("Пожалуйста, введите текст!")
    else:
        with st.spinner("Нейросеть думает..."):

            result = model(user_text)[0]
            label = result['label']
            score = result['score']

            if label == "positive":
                st.success(f"**Позитивный текст!** 🟢 (Уверенность: {score:.1%})")
            elif label == "negative":
                st.error(f"**Негативный текст!** 🔴 (Уверенность: {score:.1%})")
            else:
                st.info(f"**Нейтральный текст.** ⚪ (Уверенность: {score:.1%})")

st.markdown("---")
st.caption("Создано с помощью Streamlit и Hugging Face Transformers. [Мой GitHub](https://github.com/Gooddru)")
