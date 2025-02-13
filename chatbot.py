from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load Model & Tokenizer
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def chat():
    print("🤖 Medical Chatbot: Hello! How can I assist you today? (Type 'exit' to quit)")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("🤖 Goodbye!")
            break

        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=200)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"🤖 {response}")


if __name__ == "__main__":
    chat()
