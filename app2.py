
import pandas as pd
import streamlit as st
from srcutils import get_preprompt,format_question,hfhub
import warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide",page_title="data intrct")

st.markdown("<h1 style='text-align: center; font-weight:bold; font-family:Calibri; padding-top: 0rem;'> \
            AI Companion to Interact with Your Data</h1>", unsafe_allow_html=True)
#st.sidebar.write(":clap: :red[*Code Llama 3 model coming soon....*]")
st.sidebar.markdown('<a style="text-align: center;padding-top: 0rem;" href="mailto: usct01@gmail.com">:email:</a> Configure the Tool', unsafe_allow_html=True)

st.sidebar.caption("Development in Progress")               

available_models = {"Code Llama":"codellama/CodeLlama-34b-Instruct-hf",
                    "Mistral":"mistralai/Mistral-7B-Instruct-v0.2",
                    "Databricks":"databricks/dbrx-instruct",
                    # "ChatGPT-4": "gpt-4","ChatGPT-3.5": "gpt-3.5-turbo","GPT-3": "text-davinci-003",
                    # "GPT-3.5 Instruct": "gpt-3.5-turbo-instruct"
                    }
# available_models["Code Llama"]

# @st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Data loaded successfully!")
            return df 
        except Exception as e:
            st.error(f"Error loading data: {e}")

with st.sidebar:
    hf_key = st.text_input(label = ":hugging_face: HuggingFace Key:",help="Required for Code Llama", type="password")
    # First we want to choose the dataset, but we will fill it with choices once we've loaded one
    dataset_container = st.empty()
    # Add facility to upload a dataset
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df_name = uploaded_file.name.split('.')[0]
        dataset = load_data(uploaded_file)
        st.write(f"Loaded dataset: {df_name}")
    else:
        st.write("No dataset loaded yet. Please upload a CSV file.")
    # Check boxes for model choice
    model_type = st.radio(":brain: Choose your Model:",available_models.keys())
    
 # Text area for query
question = st.text_area(f"What would you like to know from your dataset?",height=10)
get_answer = st.button("Answer Me....")

# Query the LLM and show the results
if get_answer:
    api_keys_entered = True
    if not hf_key.startswith('hf_'):
        st.error("Please enter a valid HuggingFace API key.")
        api_keys_entered = False

    if api_keys_entered:
        primer1,primer2 = get_preprompt(dataset,df_name) 
        st.subheader(model_type)
        model_type = available_models[model_type]
        try:
            question_to_ask = format_question(primer1, primer2, question, model_type)   
            answer=""
            answer = hfhub(question_to_ask,model_type,hf_key)
            answer= answer.split('"""')[-1]
            st.code(answer,language='python') 
            plot_area = st.empty()
            # st.pyplot(exec(answer))
            plot_area.pyplot(exec(answer))     
        except Exception as e:
                st.error("Unfortunately the code generated from the model contained errors and was unable to execute.")

if uploaded_file is not None:
    st.subheader(df_name)
    st.dataframe(dataset,hide_index=True)

# Hide menu and footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
