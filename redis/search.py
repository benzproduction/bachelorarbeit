import streamlit as st
import openai
from helpers import get_env
from database import get_redis_connection, get_redis_results2
from config import INDEX_NAME, COMPLETIONS_MODEL
import tiktoken

API_KEY, RESOURCE_ENDPOINT = get_env("azure-openai")

openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = RESOURCE_ENDPOINT
openai.api_version = "2022-12-01"


tokenizer = tiktoken.get_encoding("cl100k_base")

# initialise Redis connection

client = get_redis_connection(password="weak")
INDEX_NAME = "real_estate_index"
### SEARCH APP

st.set_page_config(
    page_title="Streamlit Search - Demo",
    page_icon=":robot:"
)

st.title('Real Estate Documents Search')
st.subheader("Search for any questions about the test documents you have")

prompt = st.text_input("Enter your search here","", key="input")

if st.button('Submit', key='generationSubmit'):
    result_df = get_redis_results2(client,prompt,INDEX_NAME, top_k=8)
    # if the result_df is an empty dataframe, return a message to the user
    if result_df.empty:
        st.write("Beep Boop, I don't know the answer to that.")
    else:
        # Build a prompt to provide the original query, the result and ask to summarise for the user
        summary_prompt = '''<|im_start|>system \n
You are an intelligent assistant helping an employee with general questions regarding a for them unknown knowledge base. Be brief in your answers.
Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. 
Do not generate answers that don't use the sources below. Answer the question in the language of the employees question.
If it aims to get a yes/no response, start with a yes or no.
For tabular information return it as an html table. Do not return markdown format.
Each source has a name followed by colon and the actual information ending with a semicolon, always include the source name for each fact you use in the response.
Use square brackets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf].
The employee is asking: {injected_prompt}
Sources:{sources}
<|im_end|>
'''

        # loop over the results and format the result string in the format described above
        for i, row in result_df.iterrows():
            result_df.loc[i,'result'] = f"{row['filename']}: {row['result']};"
        # combine all returned sources into one string
        result_string = result_df['result'].str.cat(sep="\n\n")

        summary_prepped = summary_prompt.format(
                injected_prompt=prompt,
                sources=result_string
            )
        print("TOKENLEN:",len(tokenizer.encode(summary_prepped)))
        # print(summary_prepped)
        summary = openai.Completion.create(
                engine=COMPLETIONS_MODEL,
                prompt=summary_prepped,
                temperature=0.0,
                max_tokens=300,
                n=1,
                stop=["<|im_end|>", "<|im_start|>"]
            )
        
            # Response provided by GPT-3
        st.write(summary['choices'][0]['text'])