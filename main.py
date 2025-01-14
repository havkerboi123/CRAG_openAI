from openai import OpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
import os
os.environ["TAVILY_API_KEY"] = "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"



#Retriving from the vectorStore
CHROMA_PATH="/Users/mhmh/Desktop/crag/"
embedding_function = OpenAIEmbeddings(api_key="")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function,collection_name="wiki")

retriever = db.as_retriever(search_type="similarity_score_threshold",search_kwargs={"k": 3,"score_threshold": 0.3})






client = OpenAI(api_key="")

def rewrite_query(user_query):
    SYS_PROMPT = """Act as a question re-writer and perform the following task:
                 - Convert the following input question to a better version that is optimized for web search.
                 - When re-writing, look at the input question and try to reason about the underlying semantic intent / meaning.
             """
    response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"Reswrite the following user-query : {user_query}"},
                        
                    ]}
                ],
                temperature=0.0,
            )
    result = response.choices[0].message.content
    return result


def grade_documents(docs, user_query):
    
    SYS_PROMPT = """You are an expert grader assessing relevance of a retrieved document to a user question.
    Follow these instructions for grading:
    
    - Your grade should be either 'yes' or 'no' to indicate whether the document is relevant to the question or not.
    - If you can provide an accurate answer to the user's query from provided docs then grade 'yes' else grade 'no'.
    """
    
    #format the docs
    docs_text = "\n".join([str(doc) for doc in docs])
    
    response = client.chat.completions.create(
        model="gpt-4o",  
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": f"Review the docs and tell in 'yes' or 'no'. ONLY ANSWER YES IF THE ANSWER TO USER'S QUERY IS IN THE CONTEXT PROVIDED TO YOU:\n{user_query}\n\nThe documents:\n{docs_text}"}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content


def web_search(user_query):
    search = TavilySearchResults(tavily_api_key="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", max_results=3, search_depth='advanced')

    searched_docs=search.invoke(user_query)
    docs_text = "\n".join([str(doc) for doc in searched_docs])
    return docs_text
  


def basic_rag(user_query,final_docs):
     
    SYS_PROMPT = """You are an assistant for question-answering tasks.

            Use the following pieces of retrieved context to answer the question.
            If no context is present or if you don't know the answer, just say that you don't know the answer.
            Do not make up the answer unless it is there in the provided context.
            Give a detailed answer and to the point answer with regard to the question."""
    
    
    
    response = client.chat.completions.create(
        model="gpt-4o",  
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": f"Use the context provided to answer the user's query:\n{user_query}\n\nContext:\n{final_docs}"}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content


user_quer="who is the ceo of apple"
retrived_docs = retriever.invoke(user_quer)

needs_search = grade_documents(retrived_docs,user_quer)


if needs_search=='no':
    
    new_query = rewrite_query(user_quer)
    search_docs= web_search(new_query)
    result = basic_rag(user_quer,search_docs)
else:
    result=basic_rag(user_quer,retrived_docs)    

print(result)








