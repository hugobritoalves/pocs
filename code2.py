"""
title: AWS Bedrock RAG Pipeline
author: Hugo
date: 2024-10-09
version: 1.9
license: MIT
description: A pipeline for performing Retrieve-and-Generate (RAG) using AWS Bedrock Agent Runtime with additional parameters.
requirements: boto3
environment_variables: AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION_NAME, KNOWLEDGE_BASE_ID, BEDROCK_MODEL_ID
"""

import logging
import os
import boto3
import uuid
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
        BEDROCK_MODEL_ID: str = "anthropic.claude-2"  # Modelo padrão
        DEFAULT_NUMBER_OF_RESULTS: int = 3  # Número padrão de resultados
        DEFAULT_PROMPT_TEMPLATE: str = ""  # Template de prompt padrão

    def __init__(self):
        # Nome da pipeline
        self.name = "Ulife Code 2"  # Nome personalizado

        # Configuração das válvulas e credenciais
        self.valves = self.Valves(
            AWS_ACCESS_KEY=os.getenv("AWS_ACCESS_KEY", ""),
            AWS_SECRET_KEY=os.getenv("AWS_SECRET_KEY", ""),
            AWS_REGION_NAME=os.getenv("AWS_REGION_NAME", "us-east-1"),
            KNOWLEDGE_BASE_ID=os.getenv("KNOWLEDGE_BASE_ID", ""),
            BEDROCK_MODEL_ID=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-2"),
            DEFAULT_NUMBER_OF_RESULTS=int(os.getenv("DEFAULT_NUMBER_OF_RESULTS", 3)),
            DEFAULT_PROMPT_TEMPLATE=os.getenv("DEFAULT_PROMPT_TEMPLATE", ""),
        )

        # Configurando cliente do Bedrock Agent Runtime
        self.bedrock_agent_runtime = boto3.client(
            "bedrock-agent-runtime",
            aws_access_key_id=self.valves.AWS_ACCESS_KEY,
            aws_secret_access_key=self.valves.AWS_SECRET_KEY,
            region_name=self.valves.AWS_REGION_NAME,
        )

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict,
        __user__: dict = None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> Union[str, dict]:
        # Verificar se a consulta do usuário foi fornecida
        if not user_message:
            logging.error("Nenhuma consulta fornecida.")
            return {"status": "error", "message": "Nenhuma consulta fornecida."}

        # Pop de system message para ajustar o contexto
        system_message, messages = pop_system_message(messages)

        # Recuperar sessionId do body ou gerar um novo
        session_id = body.get("sessionId")
        if not session_id:
            # Gerar um sessionId único
            session_id = str(uuid.uuid4())

        # Obter numberOfResults do body ou usar o padrão
        number_of_results = body.get("numberOfResults", self.valves.DEFAULT_NUMBER_OF_RESULTS)

        # Obter promptTemplate do body ou usar o padrão
        prompt_template = body.get("promptTemplate", self.valves.DEFAULT_PROMPT_TEMPLATE)

        # Construir o payload para o Retrieve-and-Generate
        try:
            payload = {
                "input": {"text": user_message},
                "retrieveAndGenerateConfiguration": {
                    "type": "KNOWLEDGE_BASE",
                    "knowledgeBaseConfiguration": {
                        "knowledgeBaseId": self.valves.KNOWLEDGE_BASE_ID,
                        "modelArn": f"arn:aws:bedrock:{self.valves.AWS_REGION_NAME}::foundation-model/{self.valves.BEDROCK_MODEL_ID}",
                        "generationConfiguration": {},
                        "retrievalConfiguration": {
                            "vectorSearchConfiguration": {
                                "numberOfResults": number_of_results,
                            }
                        },
                    }
                },
                "sessionId": session_id,
            }

            # Incluir promptTemplate se fornecido
            if prompt_template:
                payload["retrieveAndGenerateConfiguration"]["knowledgeBaseConfiguration"]["generationConfiguration"]["promptTemplate"] = {
                    "textPromptTemplate": prompt_template
                }

            # Chamada direta para obter o texto completo
            return self.get_completion(payload)

        except Exception as e:
            logging.error(f"Erro ao processar a consulta RAG: {e}")
            return {"status": "error", "message": str(e)}

    def get_completion(self, payload: dict) -> str:
        try:
            # Fazendo a chamada ao Bedrock Agent Runtime
            response = self.bedrock_agent_runtime.retrieve_and_generate(**payload)

            # Extrair o texto gerado da resposta
            candidates = response.get('candidates', [])
            if candidates:
                generated_text = candidates[0]['output']['message']['content'][0]['text']
                return generated_text
            else:
                return "Nenhuma resposta foi gerada."

        except Exception as e:
            logging.error(f"Erro ao obter a resposta: {e}")
            return f"Error: {e}"
