# %%
import pandas as pd
import requests
from elasticsearch import Elasticsearch
from tqdm.auto import tqdm
from openai import OpenAI
from decouple import config

es = Elasticsearch("http://localhost:9200")
# client = OpenAI(
#     base_url='http://localhost:11434/v1/',
#     api_key='ollama',
# )
client = OpenAI(
    api_key=config("api_key"),
    base_url="https://api.aimlapi.com",
)

# %%


def get_documents():
    docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()
    # print(documents_raw)
    documents = []

    for course in documents_raw:
        course_name = course['course']
        # print(course_name)
        for doc in course['documents']:
            doc['course'] = course_name
            documents.append(doc)
    return documents

# %%


def setup_index(index_name, documents):
    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "section": {"type": "text"},
                "question": {"type": "text"},
                "course": {"type": "keyword"}
            }
        }
    }
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=index_settings)

    for doc in tqdm(documents):
        es.index(index=index_name, document=doc)
    return index_name

# %%


def retrieval(query, index_name='course-questions', max_results=5):
    search_query = {
        "size": max_results,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": "data-engineering-zoomcamp"
                    }
                }
            }
        }
    }

    response = es.search(index=index_name, body=search_query)
    top_n_matched_questions = [hit['_source']
                               for hit in response['hits']['hits']]
    return top_n_matched_questions


# %%
context_template = """
Section: {section}
Question: {question}
Answer: {text}
""".strip()

prompt_template = """
You're a course teaching assistant.
Answer the user QUESTION based on CONTEXT - the documents retrieved from our FAQ database.
Don't use other information outside of the provided CONTEXT.  

QUESTION: {user_question}

CONTEXT:

{context}
""".strip()


def build_context(documents):
    context_result = ""

    for doc in documents:
        doc_str = context_template.format(**doc)
        context_result += ("\n\n" + doc_str)

    return context_result.strip()


def build_prompt(user_question, documents):
    context = build_context(documents)
    prompt = prompt_template.format(
        user_question=user_question,
        context=context
    )
    return prompt


def ask_openai(prompt, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model='mistralai/Mistral-7B-Instruct-v0.2',
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    return answer


def qa_bot(user_question):
    context_docs = retrieval(user_question)
    # print(context_docs)
    prompt = build_prompt(user_question, context_docs)
    # print(prompt)
    answer = ask_openai(prompt, "phi3")
    print(answer)
    return answer


print("p")

# %%
# we need to include the documents in the elastic_search, only after that we can run rest of the processes of prompting


def run():
    documents = get_documents()
    index_name = setup_index("course-questions", documents)


# %%
if __name__ == "__main__":
    run()
    qa_bot("what can i learn from the course?")

# %%
# run()
# execute only at the start of inserting docs into elastic_search

# %%
