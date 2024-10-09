"""
title: AWS Bedrock RAG Pipeline
author: Hugo
date: 2024-10-09
version: 1.5
license: MIT
description: A pipeline for performing Retrieve-and-Generate (RAG) using AWS Bedrock Agent Runtime with Claude 3 Haiku, including stream response.
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
        AWS_REGION_NAME: str = ""
        KNOWLEDGE_BASE_ID: str = ""
        BEDROCK_MODEL_ID: str = "anthropic.claude-3-haiku-20240307-v1:0"  # Modelo padrão

    def __init__(self):
        # Configuração das válvulas e credenciais
        self.valves = self.Valves(
            AWS_ACCESS_KEY=os.getenv("AWS_ACCESS_KEY", ""),
            AWS_SECRET_KEY=os.getenv("AWS_SECRET_KEY", ""),
            AWS_REGION_NAME=os.getenv("AWS_REGION_NAME", "us-east-1"),
            KNOWLEDGE_BASE_ID=os.getenv("KNOWLEDGE_BASE_ID", ""),
            BEDROCK_MODEL_ID=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"),
        )

        # Configurando cliente do Bedrock Agent Runtime
        self.bedrock_agent_runtime = boto3.client(
            "bedrock-agent-runtime",
            aws_access_key_id=self.valves.AWS_ACCESS_KEY,
            aws_secret_access_key=self.valves.AWS_SECRET_KEY,
            region_name=self.valves.AWS_REGION_NAME,
        )

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # Verificar se a consulta do usuário foi fornecida
        if not user_message:
            logging.error("Nenhuma consulta fornecida.")
            return {"status": "error", "message": "Nenhuma consulta fornecida."}

        # Pop de system message para ajustar o contexto
        system_message, messages = pop_system_message(messages)

        try:
            # Construir o payload para o Retrieve-and-Generate
            payload = {
                "input": {"text": user_message},
                "retrieveAndGenerateConfiguration": {
                    "type": "KNOWLEDGE_BASE",
                    "knowledgeBaseConfiguration": {
                        "knowledgeBaseId": self.valves.KNOWLEDGE_BASE_ID,
                        "modelArn": f"arn:aws:bedrock:{self.valves.AWS_REGION_NAME}::foundation-model/{self.valves.BEDROCK_MODEL_ID}"
                    }
                }
            }

            # Verificar se o body contém a chave para stream
            if body.get("stream", False):
                # Se stream estiver ativado, processar resposta via stream
                return self.stream_response(payload)
            else:
                # Caso contrário, chamada direta para obter o texto completo
                return self.get_completion(payload)

        except Exception as e:
            logging.error(f"Erro ao processar a consulta RAG: {e}")
            return {"status": "error", "message": str(e)}

    def stream_response(self, payload: dict) -> Generator:
        try:
            # Chamando API com suporte a streaming
            streaming_response = self.bedrock_agent_runtime.retrieve_and_generate_stream(**payload)
            for chunk in streaming_response["stream"]:
                if "contentBlockDelta" in chunk:
                    # Retornar os pedaços da resposta à medida que são gerados
                    yield chunk["contentBlockDelta"]["delta"]["text"]

        except Exception as e:
            logging.error(f"Erro no streaming: {e}")
            yield f"Error: {e}"

    def get_completion(self, payload: dict) -> str:
        try:
            # Chamada regular para obter a resposta completa
            response = self.bedrock_agent_runtime.retrieve_and_generate(**payload)
            return response['output']['text']

        except Exception as e:
            logging.error(f"Erro ao obter a resposta: {e}")
            return f"Error: {e}"
