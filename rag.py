"""
title: AWS Bedrock RAG Pipeline
author: Seu Nome
date: 2024-10-09
version: 1.0
license: MIT
description: A pipeline for generating text using the AWS Bedrock Retrieve-and-Generate API (RAG) with Claude 3 Haiku.
requirements: boto3
environment_variables: AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION_NAME, KNOWLEDGE_BASE_ID, BEDROCK_MODEL_ID
"""

import os
import logging
from typing import List, Union, Generator, Iterator

import boto3
from pydantic import BaseModel

from utils.pipelines.main import pop_system_message

class Pipeline:
    class Valves(BaseModel):
        AWS_ACCESS_KEY: str = ""
        AWS_SECRET_KEY: str = ""
        AWS_REGION_NAME: str = ""
        KNOWLEDGE_BASE_ID: str = ""
        BEDROCK_MODEL_ID: str = "anthropic.claude-v3-haiku"  # Modelo padrão

    def __init__(self):
        self.type = "manifold"
        self.name = "Bedrock RAG Pipeline"

        self.valves = self.Valves(
            **{
                "AWS_ACCESS_KEY": os.getenv("AWS_ACCESS_KEY", ""),
                "AWS_SECRET_KEY": os.getenv("AWS_SECRET_KEY", ""),
                "AWS_REGION_NAME": os.getenv("AWS_REGION_NAME", "us-east-1"),
                "KNOWLEDGE_BASE_ID": os.getenv("KNOWLEDGE_BASE_ID", ""),
                "BEDROCK_MODEL_ID": os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-v3-haiku"),
            }
        )

        self.bedrock_agent_runtime = boto3.client(
            "bedrock-agent-runtime",
            aws_access_key_id=self.valves.AWS_ACCESS_KEY,
            aws_secret_access_key=self.valves.AWS_SECRET_KEY,
            region_name=self.valves.AWS_REGION_NAME
        )

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        pass

    async def on_valves_updated(self):
        self.bedrock_agent_runtime = boto3.client(
            "bedrock-agent-runtime",
            aws_access_key_id=self.valves.AWS_ACCESS_KEY,
            aws_secret_access_key=self.valves.AWS_SECRET_KEY,
            region_name=self.valves.AWS_REGION_NAME
        )

    def pipelines(self) -> List[dict]:
        return self.get_models()

    def get_models(self):
        # Aqui, você pode listar os modelos disponíveis se desejar
        # Neste exemplo, retornamos apenas o modelo definido
        return [
            {
                "id": self.valves.BEDROCK_MODEL_ID,
                "name": f"Bedrock RAG Model: {self.valves.BEDROCK_MODEL_ID}",
            },
        ]

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # Certifique-se de que o KNOWLEDGE_BASE_ID está definido
        if not self.valves.KNOWLEDGE_BASE_ID:
            return "Error: KNOWLEDGE_BASE_ID is not set. Please set it in the environment variables."

        # Obtenha a mensagem do usuário
        system_message, messages = pop_system_message(messages)
        logging.info(f"pop_system_message: {messages}")

        # Construir a entrada para o retrieve_and_generate
        query = user_message
        try:
            response = self.bedrock_agent_runtime.retrieve_and_generate(
                input={'text': query},
                retrieveAndGenerateConfiguration={
                    'type': 'KNOWLEDGE_BASE',
                    'knowledgeBaseConfiguration': {
                        'knowledgeBaseId': self.valves.KNOWLEDGE_BASE_ID,
                        'modelArn': f'arn:aws:bedrock:{self.valves.AWS_REGION_NAME}::foundation-model/{self.valves.BEDROCK_MODEL_ID}'
                    }
                },
            )
            generated_text = response['output']['text']
            return generated_text
        except Exception as e:
            logging.error(f"Error in retrieve_and_generate: {e}")
            return f"Error: {e}"
