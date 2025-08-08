from typing import List, Sequence
from dotenv import load_dotenv 
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph  # Updated import 
from chains import generation_chain, reflection_chain 

load_dotenv()