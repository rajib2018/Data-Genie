import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

st.title("🤖 Data Genie 🧞‍♂️")
st.subheader("Ask your data anything (within reason 😉)")

st.sidebar.title("✨ Magic Data Corner ✨")
st.sidebar.write("This is where the data magic happens... or just some cool info!")

# Add a small, relevant image or GIF (replace with a relevant image URL or file path)
# Example using a placeholder image URL
st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-light.png", caption="Unlocking data secrets!", width=200)


uploaded_file = st.file_uploader(
    "Upload your data file (CSV, Excel, etc.)", type=["csv", "xls", "xlsx"]
)

question = st.text_input("Ask a question about your data:")

if uploaded_file is not None:
    try:
        # Determine the file type
        file_extension = uploaded_file.name.split(".")[-1].lower()

        if file_extension == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_extension in ["xls", "xlsx"]:
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type.")
            df = None

        if df is not None:
            st.success("File successfully loaded! Ready to sprinkle some data magic.")
            # Optional: display the first few rows of the dataframe
            # st.dataframe(df.head())

    except Exception as e:
        st.error(f"Error reading file: {e}")
        df = None
else:
    df = None # Ensure df is None if no file is uploaded

# Select a different, publicly available pre-trained language model
model_name = "gpt2"

@st.cache_resource
def load_model_and_tokenizer(model_name):
    """Loads the Hugging Face model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# Load model only if file and question are present to avoid unnecessary loading
if uploaded_file is not None and question and df is not None:
    tokenizer, model = load_model_and_tokenizer(model_name)

    try:
        # Convert DataFrame to a string representation or a summary
        # Simple representation for demonstration
        data_representation = df.to_string() # Or df.head().to_string() for larger files

        # Combine data and question into a prompt
        prompt = f"Given the following data:\n{data_representation}\n\nAnswer the question: {question}"

        # Use a pipeline for easier generation
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

        # Generate the answer, potentially handling truncation
        # Adjust max_length and truncation as needed
        answer = generator(prompt, max_length=512, truncation=True)[0]['generated_text']

        # Add animations before displaying the answer
        st.balloons()
        st.snow()

        # Use markdown for styling the question and answer
        st.markdown(f"**Your Quirky Question:** *{question}*")
        st.markdown(f"**The Data Genie's Wisdom:**\n{answer}")

    except Exception as e:
        st.error(f"Error generating answer: {e}")
