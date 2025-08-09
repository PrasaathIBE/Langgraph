import os
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from schema import AnswerQuestion, ReviseAnswer
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.messages import HumanMessage

pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])

load_dotenv()


# Actor Agent Prompt
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert AI researcher.
            Current time: {time}

            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. After the reflection, **list 1-3 search queries seperately** for researching improvements.
            Do not include them inside the reflection.
            """,
        ),
        MessagesPlaceholder(variable_name="message"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed answer of at least 250 words. Do not answer in less than 250 words. If your answer is shorter, continue writing until you reach the required length."
)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",
                             api_key=os.getenv("GOOGLE_API_KEY"))

first_responder_chain = first_responder_prompt_template | llm.bind_tools(tools=[AnswerQuestion],tool_choice='AnswerQuestion') 

validator =  PydanticToolsParser(tools=[AnswerQuestion])

# Revisor Agent Prompt

revisor_instructions = """Revise your previous answer using the new 
information.
 - You should use the previous critique to add important information to 
 your answer.
  - You MUST include numerical citations in your revised answer to ensure 
  it can be verified.
  - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
    - [1] https://example.com 
    - [2] https://example2.com
  - You must use the previous critique to remove superfluous 
  information from your answer and make SURE it is not more than 250 words.
  """

revisor_chain = actor_prompt_template.partial(
    first_instruction=revisor_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice='ReviseAnswer')

response = first_responder_chain.invoke({
    "message": [HumanMessage(content="write me a blog post on how small businesses can leverage AI to grow")]
}) 

print(response)

