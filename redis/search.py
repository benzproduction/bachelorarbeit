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

st.title('Real Estate Test Documents Search')
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
        self_eval_prompt = '''<|im_start|>system \n
You are an intelligent assistant helping an employee with general questions regarding a for them unknown knowledge base.
You are given a question and a list of sources. Each source has a name followed by colon and the actual information ending with a semicolon
You now need to evaluate if the sources provided are sufficient to answer the question. 
Evalute based on the following criteria:
- Does the sources contain all aspects of the question?
- Does the sources contain all relevant information?
- If you need to compare something, do you have sources for both sides?

Are the sources sufficient to answer the question? Print only the single character "Y" or "N" (without quotes or punctuation) on its own line. At the end, repeat just the letter again by itself on a new line.
The employee is asking: {injected_prompt}\n
Sources:
{sources}
'''
## If the sources are not sufficient, say "False" and provide a list of exactly 3 prompts that specifically ask for the missing information.
## The False answer should be in the format: "False: [prompt1]; [prompt2]; [prompt3];"

# possible additional prompt sentences:
# For example, if the question is \"What color is the sky?\" and one of the information sources says \"info123: the sky is blue whenever it's not cloudy\", then answer with \"The sky is blue [info123]\". 

        # loop over the results and format the result string in the format described above
        for i, row in result_df.iterrows():
            result_df.loc[i,'result'] = f"{row['filename']}: {row['result']};"
        # combine all returned sources into one string
        result_string = result_df['result'].str.cat(sep="\n\n")

        # self_eval_prompt = self_eval_prompt.format(
        #     injected_prompt=prompt,
        #     sources=result_string
        # )
        # self_eval = openai.Completion.create(
        #     engine=COMPLETIONS_MODEL,
        #     prompt=self_eval_prompt,
        #     temperature=0.5,
        #     max_tokens=156, 
        #     n=1, 
        #     stop=["<|im_end|>", "<|im_start|>"])
        # self_eval = self_eval['choices'][0]['text']
        # print(self_eval)
        
        # if "N" in self_eval:
        #     st.write("I don't know the answer to that.")

        # else:
        #     # inject the prompt and the result string into the summary prompt
            
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

            # Header to give the user feedback what input led to the result
            # st.subheader("Prompt used to generate this output:")

            # # Display the prompt used to generate the result
            # st.write(summary_prepped)

            # Result string Output
            #st.write(result_string)

            # Option to display raw table instead of summary from GPT-3
            #st.table(result_df)