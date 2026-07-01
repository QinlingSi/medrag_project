from langchain_community.llms import Ollama

llm = Ollama(model="deepseek-r1:7b")
response = llm.invoke("请简要解释什么是2型糖尿病，用中文回答")
print("=== 模型返回 ===")
print(response)
