import streamlit as st
import os 

#from langchain.agents import create_csv_agent
#from langchain_experimental.agents import create_csv_agent
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv



def main():

    load_dotenv()

    st.set_page_config(page_title = "Ask your CSV")
    st.header("ASK YOUR CSV")

    user_csv = st.file_uploader("Upload your CSV file", type="csv")

    if user_csv is not None:
        user_question = st.text_input("Ask a question about your CSV")

        llm = ChatGoogleGenerativeAI(temperature=0, model='gemini-pro')
        agent = create_csv_agent(llm,user_csv, verbose=True)

        if user_question is not None and user_question != "":
            response = agent.run(user_question)
            st.write(response)
            # st.write(f"your question was: {user_question} ")


if __name__ == "__main__":
    main()

