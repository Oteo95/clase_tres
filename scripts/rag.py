from langchain.docstore.document import Document
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings

df = pd.read_csv("cities.csv")
docs = []
for idx, i in df.iterrows():
    docs.append(
        Document(
            page_content=f"{i["description"]}",
            metadata={"source":"local", "city":i["city"], "description": i["description"]}
        )
    )
    docs.append(
        Document(
            page_content=f"{i["city"]}",
           metadata={"source":"local", "city":i["city"], "description": i["description"]}
        )
    )
    
db = FAISS.from_documents(docs, HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en-v1.5"))


retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

docs = retriever.get_relevant_documents("Museums")



import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
tokenize = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")


from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser


text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenize,
    task="text-generation",
    temperature=0.1,
    do_sample=True,
    max_new_tokens=500,
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)


prompt_template = """
<|system|>
Answer the question based on the context that is presented in the # Context section. If you are not able
to answer with the information in the context just say that you are unable to answer.
Just answer the question do not provie any other statement.

# Context

{context}

</system>
<|user|>
{question}
</user>
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

llm_chain = prompt | llm | StrOutputParser()

from langchain_core.runnables import RunnablePassthrough

question="Where is the eiffel tower?"


docs = retriever.get_relevant_documents(question)

context = [i.page_content for i in docs]


llm_chain.invoke({"context": context, "question": question})






prompt_template2 = """
<|system|>
Tu tarea es entender el contecto de la conversacion y evaluar si la follow up question 
esta relacionada con las respuestas y preguntas anteriores o no. 

Si la follow up question esta relacionada con las preguntas anteriores reformula la pregunta para
que sea una standalone question.

Si la follow up question no hace referencia a la conversacion anterior devuelve el mismo mensaje.

# Chat history

{chat_history}

</system>
<|user|>
{followup_question}
</user>
"""


prompt2 = PromptTemplate(
    input_variables=["chat_history", "followup_question"],
    template=prompt_template2,
)

llm_chain_standalone = prompt2 | llm | StrOutputParser()


llm_chain_standalone.invoke({"context": context, "question": question})
