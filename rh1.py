import logging
import os
import boto3
from typing import List, Union
from pydantic import BaseModel

import json

# Importando função auxiliar para pop de system message
from utils.pipelines.main import pop_system_message

class Pipeline:
    class Valves(BaseModel):
        AWS_ACCESS_KEY_ANIMA: str = ""
        AWS_SECRET_KEY_ANIMA: str = ""
        AWS_REGION_NAME_ANIMA: str = ""
        KNOWLEDGE_BASE_W3: str = ""
        BEDROCK_MODEL_ID_W3: str = "amazon.nova-lite-v1:0"  
        DEFAULT_NUMBER_OF_RESULTS_W3: int = 10            
        DEFAULT_PROMPT_TEMPLATE_W3: str = ""              

    def __init__(self):
        self.name = os.path.basename(__file__)
        self.valves = self.Valves(
            AWS_ACCESS_KEY_ANIMA=os.getenv("AWS_ACCESS_KEY_ANIMA"),
            AWS_SECRET_KEY_ANIMA=os.getenv("AWS_SECRET_KEY_ANIMA"),
            AWS_REGION_NAME_ANIMA=os.getenv("AWS_SECRET_KEY_REGION","us-east-1"),
            KNOWLEDGE_BASE_W3="SC35OARR8D",
            BEDROCK_MODEL_ID_W3="amazon.nova-lite-v1:0",
            DEFAULT_NUMBER_OF_RESULTS_W3=int(os.getenv("DEFAULT_NUMBER_OF_RESULTS_W3", 10)),
            DEFAULT_PROMPT_TEMPLATE_W3=os.getenv(
                "DEFAULT_PROMPT_TEMPLATE_W3",
                """Você é um especialista em responder perguntas baseando-se em resultados de pesquisa fornecidos. 
O usuário fornecerá uma pergunta, e sua tarefa é responder a essa pergunta usando exclusivamente as informações contidas nos resultados de pesquisa abaixo. 
Resuma as informações mais relevantes para garantir que o usuário tenha uma compreensão clara e completa. 
Se a resposta não estiver presente nos resultados da pesquisa, informe que a informação não está disponível. 
Forneça a resposta no idioma solicitado pelo usuário. Sempre dê a resposta em português.

Resultados da Pesquisa: <context> $search_results$ </context>
Pergunta do Usuário: <question> $query$ </question>"""
            ),
        )

        self.bedrock_agent_runtime = boto3.client(
            "bedrock-agent-runtime",
            aws_access_key_id=self.valves.AWS_ACCESS_KEY_ANIMA,
            aws_secret_access_key=self.valves.AWS_SECRET_KEY_ANIMA,
            region_name=self.valves.AWS_REGION_NAME_ANIMA,
        )

        self.session_id = None

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, dict]:
        if not user_message:
            logging.error("Nenhuma consulta fornecida.")
            return json.dumps({"status": "error", "message": "Nenhuma consulta fornecida."})

        system_message, messages = pop_system_message(messages)

        try:
            number_of_results = body.get("numberOfResults", self.valves.DEFAULT_NUMBER_OF_RESULTS_W3)
            prompt_template = body.get("promptTemplate", self.valves.DEFAULT_PROMPT_TEMPLATE_W3)

            payload = {
                "input": {"text": user_message},
                "retrieveAndGenerateConfiguration": {
                    "type": "KNOWLEDGE_BASE",
                    "knowledgeBaseConfiguration": {
                        "knowledgeBaseId": self.valves.KNOWLEDGE_BASE_W3,
                        "modelArn": f"arn:aws:bedrock:{self.valves.AWS_REGION_NAME_ANIMA}::foundation-model/{self.valves.BEDROCK_MODEL_ID_W3}",
                        "retrievalConfiguration": {
                            "vectorSearchConfiguration": {
                                "numberOfResults": number_of_results,
                            }
                        }
                    }
                }
            }

            if prompt_template:
                payload["retrieveAndGenerateConfiguration"]["knowledgeBaseConfiguration"]["generationConfiguration"] = {
                    "promptTemplate": {
                        "textPromptTemplate": prompt_template
                    }
                }

            return self.get_completion(model_id, payload)
        except Exception as e:
            logging.error(f"Erro ao processar a consulta RAG: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    def get_completion(self, model_id: str, payload: dict) -> str:
        try:
            response = self.bedrock_agent_runtime.retrieve_and_generate(**payload)
            output_text = response['output']['text']
            citations = response.get("citations", [])
            retrieved_references = []
            for citation in citations:
                retrieved_references.extend(citation.get("retrievedReferences", []))
            # Se houver referências, cria um texto com elas; caso contrário, retorna apenas o output_text.
            if retrieved_references:
                references_text = "\nReferências:\n" + "\n".join(str(ref) for ref in retrieved_references)
            else:
                references_text = ""
            return output_text + references_text
        except Exception as e:
            logging.error(f"Erro ao obter a resposta: {e}")
            return f"Error: {e}"
