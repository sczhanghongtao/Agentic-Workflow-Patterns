from src.config.logging import logger
from src.projects.medical_coding.agent import Agent
from src.llm.generate import ResponseGenerator
from src.commons.message import Message
import pandas as pd
import json
from src.utils.io import append_to_json_file
class Enricher(Agent):

    def __init__(self, name: str):
        super().__init__(name)
        self.path = "data/projects/medical_coding/cpt_codes/"
        self.documents = (self.process_cpt_codes(self.path + "cpt_codes_unique.txt") 
                          + self.process_cpt_codes(self.path + "procedure_codes_for_neurosurgery.xlsx"))

        self.response_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": {
                     "type": "array",
                     "items": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string"},
                            "description": {"type": "string"},
                            "extended_description": {"type": "string"}
                        },
                        "required": ["code", "description", "extended_description"]
                     }
                }
            }
        }
        
        self.system_instruction = """
        You are a medical coding expert. You are given a list of CPT codes and their corresponding descriptions. 
        for each code, you need to extend the description into a more detailed description for the procedure, with as many keywords that are relevant to the given information as possible so that the extended description can be used to search for the procedure in a database for medical coding and billing purposes. the extended description for each code should be around 100 words.

        the output should be a list of dictionaries, and its structure should be following the schema below. return the output in json format only.

        response_schema: {response_schema}
        
        """.format(response_schema=self.response_schema)

        logger.info(f"{self.name} initialized.")
    
    @staticmethod
    def process_cpt_codes(file_path: str) -> list:
        """
        Process CPT codes from either a text file or Excel file into a list of documents.
        
        Args:
            file_path (str): Path to the CPT codes file (either .txt or .xlsx)
            
        Returns:
            list: List of dictionaries containing the processed CPT codes
        """
        documents = []
        
        if file_path.endswith('.txt'):
            with open(file_path, 'r') as file:
                for line in file:
                    # Skip empty lines
                    if not line.strip():
                        continue
                        
                    # Split the line by tabs
                    parts = line.strip().split('\t')
                    
                    if len(parts) >= 2:
                        code = parts[0].strip()
                        description = " ".join(parts[1:]).strip()
                        document = {
                            # 'text': description,
                            'metadata': {
                                'source': file_path,
                                'code': code,
                                'description': description.lstrip().lstrip('0123456789').rstrip().rstrip(';')
                            }
                        }
                        if document not in documents:
                            documents.append(document)
        
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
            
            # Assuming the Excel file has 'code' and 'description' columns
            for _, row in df.iterrows():
                documents.append({
                    # 'text': str(row['CPT Description']),
                    'metadata': {
                        'source': file_path,
                        'code': str(row['CPT']),
                        'description': str(row['CPT Description'])
                    }
                })
        
        else:
            raise ValueError("Unsupported file format. Please provide either a .txt or .xlsx file")

        return documents
    def process(self,chunksize: int = 10):
        response_generator = ResponseGenerator(llm_type="deepseek")

        for i in range(0, len(self.documents), chunksize):
            logger.info(f"Enriching chunk {i} of {len(self.documents)}")
            chunk = self.documents[i:i+chunksize]
            contents = ""
            for j in range(len(chunk)):
                contents += f"code {j}: {chunk[j]['metadata']['code']} description {j}: {chunk[j]['metadata']['description']}\n".format(j=j)

            response = response_generator.generate_response(
                model_name="deepseek-chat",
                system_instruction=self.system_instruction,
                contents=[contents],
                response_schema={'type': 'json_object'}
            )
            response = response["response"]
            validated_response = self.validate_response(chunk, response)
            append_to_json_file(self.path + "cpt_codes_enriched.json", validated_response, replace=False)
            
    
    def validate_response(self, documents: list, response: list) -> list:
        """
        Validate the response from the response generator.
        """
        try:
            assert len(documents) == len(response)
        except AssertionError:
            raise ValueError("Length mismatch between documents and response")
        sorted_response = sorted(response, key=lambda x: x['code'])
        sorted_documents = sorted(documents, key=lambda x: x['metadata']['code'])
        for i in range(len(sorted_response)):
            if sorted_documents[i]['metadata']['code'] != sorted_response[i]['code']:
                raise ValueError("Code mismatch for {original_code} and {response_code}".format(original_code=sorted_documents[i]['metadata']['code'], response_code=sorted_response[i]['code']))
            if sorted_documents[i]['metadata']['description'] != sorted_response[i]['description']:
                logger.warning("Description mismatch for {original_code} and {response_code}".format(original_code=sorted_documents[i]['metadata']['code'], response_code=sorted_response[i]['code']))
                logger.warning("original description: {original_description}; response description: {response_description}".format(original_description=sorted_documents[i]['metadata']['description'], response_description=sorted_response[i]['description']))
                # raise ValueError(f"Description mismatch for {documents[i]['metadata']['code']} and {response[i]['code']}")
            sorted_documents[i]['text'] = sorted_response[i]['extended_description']
        return sorted_documents

if __name__ == "__main__":
    enricher = Enricher("enricher")
    enricher.process()
