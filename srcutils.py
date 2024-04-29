# helper functions
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

def hfhub(question_to_ask,model_type,alt_key):
    # Hugging Face model
    llm = HuggingFaceHub(huggingfacehub_api_token = alt_key, repo_id=model_type, model_kwargs={"temperature":0.001, "max_new_tokens":500})
    llm_prompt = PromptTemplate.from_template(question_to_ask)
    llm_chain = LLMChain(llm=llm,prompt=llm_prompt)
    llm_response = llm_chain.predict()
    llm_response = format_response(llm_response)
    return llm_response

def get_preprompt(df_dataset,df_name):
    preprompt_desc = f"Use a dataframe called {df_name} with columns '" + "','".join(str(x) for x in df_dataset.columns) + "'. "
    for i in df_dataset.columns:
        if len(df_dataset[i].drop_duplicates()) < 20 and df_dataset.dtypes[i]=="O":
            preprompt_desc = preprompt_desc + "\nThe column '" + i + "' has categorical values '" + \
                "','".join(str(x) for x in df_dataset[i].drop_duplicates()) + "'. "
        elif df_dataset.dtypes[i]=="int64" or df_dataset.dtypes[i]=="float64":
            preprompt_desc = preprompt_desc + "\nThe column '" + i + "' is type " + str(df_dataset.dtypes[i]) + " and contains numeric values. "   
    preprompt_desc = preprompt_desc + "\nLabel the x and y axes appropriately."
    preprompt_desc = preprompt_desc + "\nAdd a title."
    preprompt_desc = preprompt_desc + "{}" # Space for additional instructions if needed
    preprompt_desc = preprompt_desc + "\nUsing Python version 3.11.6, create a script using the dataframe df to graph the following: "
    preprompt_code = "import pandas as pd\nimport matplotlib.pyplot as plt\n"
    preprompt_code = preprompt_code + "fig,ax = plt.subplots(1,1,figsize=(10,4))\n"
    preprompt_code = preprompt_code + "ax.spines['top'].set_visible(False)\nax.spines['right'].set_visible(False) \n"
    preprompt_code = preprompt_code + "df=" + df_name + ".copy()\n"
    return preprompt_desc,preprompt_code

def format_question(preprompt_desc,preprompt_code , question, model_type):
    # Fill in the model_specific_instructions variable
    instructions = ""
    if model_type == "Code Llama":
        # Code llama tends to misuse the "c" argument when creating scatter plots
        instructions = "\nDo not use the 'c' argument in the plot function, use 'color' instead and only pass color names like 'green', 'red', 'blue'."
    preprompt_desc = preprompt_desc.format(instructions)  
    # Put the question at the end of the description preprompt within quotes, then add on the code preprompt.
    return  '"""\n' + preprompt_desc + question + '\n"""\n' + preprompt_code

def format_response( res):
    # Remove the load_csv from the answer if it exists
    csv_line = res.find("read_csv")
    if csv_line > 0:
        return_before_csv_line = res[0:csv_line].rfind("\n")
        if return_before_csv_line == -1:
            # The read_csv line is the first line so there is nothing to need before it
            res_before = ""
        else:
            res_before = res[0:return_before_csv_line]
        res_after = res[csv_line:]
        return_after_csv_line = res_after.find("\n")
        if return_after_csv_line == -1:
            # The read_csv is the last line
            res_after = ""
        else:
            res_after = res_after[return_after_csv_line:]
        res = res_before + res_after
    return res
