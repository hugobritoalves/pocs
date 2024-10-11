"""
title: AWS Bedrock RAG Pipeline
author: Hugo
date: 2024-10-09
version: 3.4
license: MIT
description: A pipeline for performing Retrieve-and-Generate (RAG) using AWS Bedrock Agent Runtime with session handling.
requirements: boto3
environment_variables: AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION_NAME, KNOWLEDGE_BASE_ID, BEDROCK_MODEL_ID
"""

import logging
import os
import boto3
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel

# Importando função auxiliar para pop de system message
from utils.pipelines.main import pop_system_message

class Pipeline:
    class Valves(BaseModel):
        AWS_ACCESS_KEY: str = ""
        AWS_SECRET_KEY: str = ""
        AWS_REGION_NAME: str = "us-east-1"
        KNOWLEDGE_BASE_ID: str = ""
        BEDROCK_MODEL_ID: str = "anthropic.claude-3-haiku-20240307-v1:0"  # Modelo padrão
        DEFAULT_NUMBER_OF_RESULTS: int = 3  # Número padrão de resultados
        DEFAULT_PROMPT_TEMPLATE: str = ""  # Template de prompt padrão

    def __init__(self):
        # Nome da pipeline
        self.name = "Code 3.4"  # Nome atualizado

        # Configuração das válvulas e credenciais
        self.valves = self.Valves(
            AWS_ACCESS_KEY=os.getenv("AWS_ACCESS_KEY", ""),
            AWS_SECRET_KEY=os.getenv("AWS_SECRET_KEY", ""),
            AWS_REGION_NAME=os.getenv("AWS_REGION_NAME", "us-east-1"),
            KNOWLEDGE_BASE_ID=os.getenv("KNOWLEDGE_BASE_ID", ""),
            BEDROCK_MODEL_ID=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"),
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

        # Variável de instância para armazenar o sessionId
        self.session_id = None

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict,
        __user__: dict = None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> Union[str, Generator, Iterator]:
        # Verificar se a consulta do usuário foi fornecida
        if not user_message:
            logging.error("Nenhuma consulta fornecida.")
            return "Nenhuma consulta fornecida."

        # Pop de system message para ajustar o contexto
        system_message, messages = pop_system_message(messages)

        # Obter numberOfResults do body ou usar o padrão
        number_of_results = body.get("numberOfResults", self.valves.DEFAULT_NUMBER_OF_RESULTS)

        # Obter promptTemplate do body ou usar o padrão
        prompt_template = body.get("promptTemplate", self.valves.DEFAULT_PROMPT_TEMPLATE)

        # Verificar se já temos um sessionId na instância
        if not self.session_id:
            # Verificar se o sessionId é fornecido no body
            self.session_id = body.get("sessionId")

        # Construir o payload para Retrieve-and-Generate
        try:
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

            # Incluir o sessionId no payload, se ele existir
            if self.session_id:
                payload["sessionId"] = self.session_id

            # Incluir promptTemplate se fornecido
            if prompt_template:
                payload["retrieveAndGenerateConfiguration"]["knowledgeBaseConfiguration"]["generationConfiguration"] = {
                    "promptTemplate": {
                        "textPromptTemplate": prompt_template
                    }
                }

            # Chamada direta para obter o texto completo
            return self.get_completion(payload)

        except Exception as e:
            logging.error(f"Erro ao processar a consulta RAG: {e}")
            return f"Erro: {e}"

    def get_completion(self, payload: dict) -> str:
        try:
            # Fazendo a chamada ao Bedrock Agent Runtime
            response = self.bedrock_agent_runtime.retrieve_and_generate(**payload)

            # Extrair o sessionId gerado ou existente
            self.session_id = response.get('sessionId', self.session_id)

            # Verificar se a resposta contém o campo "output" e "text"
            if 'output' in response and 'text' in response['output']:
                return response['output']['text']
            else:
                return "Nenhuma resposta gerada ou campo 'text' ausente."

        except Exception as e:
            logging.error(f"Erro ao obter a resposta: {e}")
            return f"Erro: {e}"
