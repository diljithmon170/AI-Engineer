# llm.py
# Lightweight LLM + LangChain prompt chaining (no LLMChain dependency)

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate


# 🔹 Load a lightweight local model
model_name = "distilgpt2"  # You can change to TinyLlama or Phi if you want
print(f"🔹 Loading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 🔹 Create text-generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=80,
    temperature=0.8,
    top_p=0.9,
    do_sample=True
)

# 🔹 Wrap in LangChain
llm = HuggingFacePipeline(pipeline=pipe)


# 🔹 Define prompts
prompt_template_name = PromptTemplate(
    input_variables=["cuisine"],
    template="I want to open a hotel for {cuisine} food. Suggest a fancy and creative name for this hotel."
)

prompt_template_items = PromptTemplate(
    input_variables=["hotel_name"],
    template="Suggest some menu items for {hotel_name}. Return them as a comma-separated list."
)


# 🔹 Manual chaining logic
def generate_hotel_info(cuisine: str):
    # Step 1: Format first prompt
    prompt_1 = prompt_template_name.format(cuisine=cuisine)
    hotel_name = llm.invoke(prompt_1).strip()

    # Step 2: Format second prompt
    prompt_2 = prompt_template_items.format(hotel_name=hotel_name)
    menu_items = llm.invoke(prompt_2).strip()

    # Step 3: Return structured output
    return {
        "hotel_name": hotel_name,
        "menu_items": menu_items
    }


# 🔹 Run the chain
if __name__ == "__main__":
    print(generate_hotel_info("Italian"))