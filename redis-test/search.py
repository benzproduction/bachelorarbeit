import streamlit as st
import openai
from helpers import get_env
from database import get_redis_connection, get_redis_results2
from config import INDEX_NAME, COMPLETIONS_MODEL

API_KEY, RESOURCE_ENDPOINT = get_env("azure-openai")

openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = RESOURCE_ENDPOINT
openai.api_version = "2022-12-01"


# initialise Redis connection

client = get_redis_connection()

### SEARCH APP

st.set_page_config(
    page_title="Streamlit Search - Demo",
    page_icon=":robot:"
)

st.title('TestDocs 1 Search')
st.subheader("Search for any questions about the test documents you have")

prompt = st.text_input("Enter your search here","", key="input")

if st.button('Submit', key='generationSubmit'):
    result_df = get_redis_results2(client,prompt,INDEX_NAME)
    # if the result_df is an empty dataframe, return a message to the user
    if result_df.empty:
        st.write("Beep Boop, I don't know the answer to that.")
    else:
        # Build a prompt to provide the original query, the result and ask to summarise for the user
        summary_prompt = '''<|im_start|>system \n
You are an intelligent assistant helping an employee with general questions regarding a for them unknown knowledge base. Be brief in your answers.
Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. 
Do not generate answers that don't use the sources below. 
If asking a clarifying question to the user would help, ask the question.
For tabular information return it as an html table. Do not return markdown format.
Each source has a name followed by colon and the actual information ending with a semicolon, always include the source name for each fact you use in the response.
Use square brakets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf].
The employee is asking: {injected_prompt}\n
Sources:
{sources}
<|im_end|>
'''
# possible additional prompt sentences:
# For example, if the question is \"What color is the sky?\" and one of the information sources says \"info123: the sky is blue whenever it's not cloudy\", then answer with \"The sky is blue [info123]\". 

        # loop over the results and format the result string in the format described above
        for i, row in result_df.iterrows():
            result_df.loc[i,'result'] = f"{row['filename']}: {row['result']};"
        # combine all returned sources into one string
        result_string = result_df['result'].str.cat(sep="\n\n")
        # inject the prompt and the result string into the summary prompt
        summary_prepped = summary_prompt.format(
            injected_prompt=prompt,
            sources=result_string
        )
        summary = openai.Completion.create(
            engine=COMPLETIONS_MODEL,
            prompt=summary_prepped,
            temperature=0.7,
            max_tokens=1024, 
            n=1, 
            stop=["<|im_end|>", "<|im_start|>"])
        
        # Response provided by GPT-3
        st.write(summary['choices'][0]['text'])

        # Header to give the user feedback what input led to the result
        st.subheader("Prompt used to generate this output:")

        # Display the prompt used to generate the result
        st.write(summary_prepped)

        # Result string Output
        #st.write(result_string)

        # Option to display raw table instead of summary from GPT-3
        #st.table(result_df)