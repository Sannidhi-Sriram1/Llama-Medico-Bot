import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Streamlit UI
st.title("🤖 Llama Medico Bot")
st.write("Ask me anything about medical symptoms! (Disclaimer: Not a substitute for professional medical advice.)")

# User input
user_input = st.text_input("You:", "")

if st.button("Ask"):
    if user_input:
        input_ids = tokenizer.encode(user_input, return_tensors="pt")
        output = model.generate(input_ids, max_length=200)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        st.write("🤖 **Chatbot:**", response)
