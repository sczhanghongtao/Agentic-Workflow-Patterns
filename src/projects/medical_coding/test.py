from src.projects.medical_coding.enricher import Enricher
from src.llm.generate import ResponseGenerator
from src.utils.io import extract_json_from_response

enricher = Enricher("enricher")
response_generator = ResponseGenerator("deepseek")

chunk = enricher.documents[:20]
contents = ""
for j in range(len(chunk)):
    contents += f"code {j}: {chunk[j]['metadata']['code']} description {j}: {chunk[j]['metadata']['description']}\n".format(j=j)

response = response_generator.generate_response(
    model_name="deepseek-chat",
    system_instruction=enricher.system_instruction,
    contents=[contents],
    response_schema={'type': 'json_object'}
)

print(extract_json_from_response(response))

