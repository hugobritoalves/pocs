"""
title: Bedrock RAG Pipeline
author: Seu Nome
date: 2024-10-09
version: 1.0
license: MIT
description: A pipeline to perform RAG with AWS Bedrock in OpenWebUI.
requirements: boto3
environment_variables: AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, KNOWLEDGE_BASE_ID, BEDROCK_MODEL_ID
"""

import os
import boto3
import logging
from typing import List, Union

from pydantic import BaseModel

# Configuração de logging para facilitar a depuração
logging.basicConfig(level=logging.INFO)

class Pipeline:
    class Valves(BaseModel):
        AWS_ACCESS_KEY: str = ""
        AWS_SECRET_KEY: str = ""
        AWS_REGION: str = ""
        KNOWLEDGE_BASE_ID: str = ""
        BEDROCK_MODEL_ID: str = "anthropic.claude-v3-haiku"  # Modelo padrão

    def __init__(self):
        # Carregar configurações das válvulas
        self.valves = self.Valves(
            AWS_ACCESS_KEY=os.getenv("AWS_ACCESS_KEY", ""),
            AWS_SECRET_KEY=os.getenv("AWS_SECRET_KEY", ""),
            AWS_REGION=os.getenv("AWS_REGION", "us-east-1"),
            KNOWLEDGE_BASE_ID=os.getenv("KNOWLEDGE_BASE_ID", "KIYHMVEX9V"),
            BEDROCK_MODEL_ID=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"),
        )

        # Configuração do cliente Bedrock Agent Runtime usando as válvulas
        self.client = boto3.client(
            "bedrock-agent-runtime",
            aws_access_key_id=self.valves.AWS_ACCESS_KEY,
            aws_secret_access_key=self.valves.AWS_SECRET_KEY,
            region_name=self.valves.AWS_REGION,
        )

        # Construir o ARN do modelo
        self.model_arn = (
            f"arn:aws:bedrock:{self.valves.AWS_REGION}::foundation-model/{self.valves.BEDROCK_MODEL_ID}"
        )

    async def pipe(
        self,
        body: dict,
        __user__: dict = None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> dict:
        # Logar o conteúdo do body para análise
        logging.info(f"Conteúdo do body recebido: {body}")

        # Buscar a lista de mensagens no body
        messages = body.get("messages", [])

        # Verificar se há mensagens e extrair a última mensagem
        if messages:
            query = messages[-1].get("content", "")
        else:
            query = ""

        # Verificar se a consulta foi fornecida
        if not query:
            logging.error("Nenhuma consulta fornecida.")
            return {"status": "error", "message": "Nenhuma consulta fornecida."}

        # Realizar a chamada ao Bedrock
        try:
            response = self.client.retrieve_and_generate(
                input={"text": query},
                retrieveAndGenerateConfiguration={
                    "type": "KNOWLEDGE_BASE",
                    "knowledgeBaseConfiguration": {
                        "knowledgeBaseId": self.valves.KNOWLEDGE_BASE_ID,
                        "modelArn": self.model_arn,
                    },
                },
            )
            generated_text = response["output"]["text"]
            # Adicionar a resposta às mensagens
            body["messages"].append({"role": "assistant", "content": generated_text})
            # Retornar o body atualizado
            return body
        except Exception as e:
            logging.error(f"Erro ao processar a consulta: {e}")
            return {"status": "error", "message": str(e)}
