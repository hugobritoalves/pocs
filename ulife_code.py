"""
title: AWS Bedrock RAG Pipeline
author: Hugo
date: 2024-10-09
version: 2.1
license: MIT
description: A pipeline for performing Retrieve-and-Generate (RAG) using AWS Bedrock Agent Runtime with additional parameters.
requirements: boto3
environment_variables: AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION_NAME, KNOWLEDGE_BASE_ID, BEDROCK_MODEL_ID
"""

import logging
import os
import boto3
from typing import List, Union
from pydantic import BaseModel

# Importando função auxiliar para pop de system message
from utils.pipelines.main import pop_system_message

class Pipeline:
    class Valves(BaseModel):
        AWS_ACCESS_KEY: str = ""
        AWS_SECRET_KEY: str = ""
        AWS_REGION_NAME: str = ""
        KNOWLEDGE_BASE_ID: str = ""
        BEDROCK_MODEL_ID: str = "amazon.nova-micro-v1:0"  # Modelo padrão
        DEFAULT_NUMBER_OF_RESULTS: int = 10  # Número padrão de resultados
        DEFAULT_PROMPT_TEMPLATE: str = ""  # Template de prompt padrão

    def __init__(self):

        # Nome da pipeline
        self.name = os.path.basename(__file__)  # Nome personalizado
        
        # Configuração das válvulas e credenciais
        self.valves = self.Valves(
            AWS_ACCESS_KEY=os.getenv("AWS_ACCESS_KEY", ""),
            AWS_SECRET_KEY=os.getenv("AWS_SECRET_KEY", ""),
            AWS_REGION_NAME=os.getenv("AWS_REGION_NAME", "us-east-1"),
            KNOWLEDGE_BASE_ID=os.getenv("KNOWLEDGE_BASE_ULIFE", ""),
            BEDROCK_MODEL_ID=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"),
            DEFAULT_NUMBER_OF_RESULTS=int(os.getenv("DEFAULT_NUMBER_OF_RESULTS", 10)),
            DEFAULT_PROMPT_TEMPLATE = os.getenv("DEFAULT_PROMPT_TEMPLATE", 
                """Você é um especialista em responder perguntas baseando-se em resultados de pesquisa fornecidos. 
                O usuário fornecerá uma pergunta, e sua tarefa é responder a essa pergunta usando exclusivamente as informações contidas nos resultados de pesquisa abaixo. 
                Resuma as informações mais relevantes para garantir que o usuário tenha uma compreensão clara e completa. 
                Se a resposta não estiver presente nos resultados da pesquisa, informe que a informação não está disponível. 
                Forneça a resposta no idioma solicitado pelo usuário. Sempre dê a resposta em português.
            
                Resultados da Pesquisa: <context> $search_results$ </context>
                Pergunta do Usuário: <question> $query$ </question>"""
            ),
        )

        # Configurando cliente do Bedrock Agent Runtime
        self.bedrock_agent_runtime = boto3.client(
            "bedrock-agent-runtime",
            aws_access_key_id=self.valves.AWS_ACCESS_KEY,
            aws_secret_access_key=self.valves.AWS_SECRET_KEY,
            region_name=self.valves.AWS_REGION_NAME,
        )

        # Inicializar session_id
        self.session_id = None

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, dict]:
        # Verificar se a consulta do usuário foi fornecida
        if not user_message:
            logging.error("Nenhuma consulta fornecida.")
            return {"status": "error", "message": "Nenhuma consulta fornecida."}

        # Pop de system message para ajustar o contexto
        system_message, messages = pop_system_message(messages)

        try:
            # Obter numberOfResults do body ou usar o padrão
            number_of_results = body.get("numberOfResults", self.valves.DEFAULT_NUMBER_OF_RESULTS)

            # Obter promptTemplate do body ou usar o padrão
            prompt_template = body.get("promptTemplate", self.valves.DEFAULT_PROMPT_TEMPLATE)

            # Construir o payload para o Retrieve-and-Generate
            payload = {
                "input": {"text": user_message},
                "retrieveAndGenerateConfiguration": {
                    "type": "KNOWLEDGE_BASE",
                    "knowledgeBaseConfiguration": {
                        "knowledgeBaseId": self.valves.KNOWLEDGE_BASE_ID,
                        "modelArn": f"arn:aws:bedrock:{self.valves.AWS_REGION_NAME}::foundation-model/{self.valves.BEDROCK_MODEL_ID}",
                        "retrievalConfiguration": {
                            "vectorSearchConfiguration": {
                                "numberOfResults": number_of_results,
                            }
                        }
                    }
                }
            }

            # Incluir promptTemplate se fornecido
            if prompt_template:
                payload["retrieveAndGenerateConfiguration"]["knowledgeBaseConfiguration"]["generationConfiguration"] = {
                    "promptTemplate": {
                        "textPromptTemplate": prompt_template
                    }
                }

            # Chamada para obter o texto completo
            return self.get_completion(model_id, payload)

        except Exception as e:
            logging.error(f"Erro ao processar a consulta RAG: {e}")
            return {"status": "error", "message": str(e)}

    def get_completion(self, model_id: str, payload: dict) -> str:
        try:
            # Fazendo a chamada ao Bedrock Agent Runtime
            response = self.bedrock_agent_runtime.retrieve_and_generate(**payload)

            # Retornar apenas o texto gerado pelo modelo
            return response['output']['text']

        except Exception as e:
            logging.error(f"Erro ao obter a resposta: {e}")
            return f"Error: {e}"
