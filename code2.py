"""
title: AWS Bedrock RAG Pipeline
author: Hugo
date: 2024-10-09
version: 3.3
license: MIT
description: A pipeline for performing Retrieve-and-Generate (RAG) using AWS Bedrock Agent Runtime with session handling, returning the generated text.
requirements: boto3
environment_variables: AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION_NAME, KNOWLEDGE_BASE_ID, BEDROCK_MODEL_ID
"""

import logging
import os
import boto3
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel

# Importing auxiliary function to pop system message
from utils.pipelines.main import pop_system_message

class Pipeline:
    class Valves(BaseModel):
        AWS_ACCESS_KEY: str = ""
        AWS_SECRET_KEY: str = ""
        AWS_REGION_NAME: str = "us-east-1"
        KNOWLEDGE_BASE_ID: str = ""
        BEDROCK_MODEL_ID: str = "anthropic.claude-3"  # Default model
        DEFAULT_NUMBER_OF_RESULTS: int = 3  # Default number of results
        DEFAULT_PROMPT_TEMPLATE: str = ""  # Default prompt template

    def __init__(self):
        # Pipeline name
        self.name = "Code 3.3"  # Updated name

        # Valve configuration and credentials
        self.valves = self.Valves(
            AWS_ACCESS_KEY=os.getenv("AWS_ACCESS_KEY", ""),
            AWS_SECRET_KEY=os.getenv("AWS_SECRET_KEY", ""),
            AWS_REGION_NAME=os.getenv("AWS_REGION_NAME", "us-east-1"),
            KNOWLEDGE_BASE_ID=os.getenv("KNOWLEDGE_BASE_ID", ""),
            BEDROCK_MODEL_ID=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3"),
            DEFAULT_NUMBER_OF_RESULTS=int(os.getenv("DEFAULT_NUMBER_OF_RESULTS", 3)),
            DEFAULT_PROMPT_TEMPLATE=os.getenv("DEFAULT_PROMPT_TEMPLATE", ""),
        )

        # Configuring Bedrock Agent Runtime client
        self.bedrock_agent_runtime = boto3.client(
            "bedrock-agent-runtime",
            aws_access_key_id=self.valves.AWS_ACCESS_KEY,
            aws_secret_access_key=self.valves.AWS_SECRET_KEY,
            region_name=self.valves.AWS_REGION_NAME,
        )

        # Instance variable to store session ID
        self.session_id = None

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict,
        __user__: dict = None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> Union[str, Generator, Iterator]:
        # Verify if user query is provided
        if not user_message:
            logging.error("No query provided.")
            return "No query provided."

        # Pop system message to adjust context
        system_message, messages = pop_system_message(messages)

        # Get numberOfResults from body or use default
        number_of_results = body.get("numberOfResults", self.valves.DEFAULT_NUMBER_OF_RESULTS)

        # Get promptTemplate from body or use default
        prompt_template = body.get("promptTemplate", self.valves.DEFAULT_PROMPT_TEMPLATE)

        # Check if we already have a sessionId in the instance
        if not self.session_id:
            # Check if sessionId is provided in the body
            self.session_id = body.get("sessionId")

        # Build the payload for Retrieve-and-Generate
        try:
            payload = {
                "inputText": user_message,
                "knowledgeBaseId": self.valves.KNOWLEDGE_BASE_ID,
            }

            # Include sessionId in the payload if it exists
            if self.session_id:
                payload["sessionId"] = self.session_id

            # Include promptTemplate if provided
            if prompt_template:
                payload["textGenerationConfig"] = {
                    "prompt": prompt_template
                }

            # Include retrieval configuration
            payload["retrievalConfig"] = {
                "numberOfResults": number_of_results,
            }

            # Make the direct call to get the complete text
            generated_text = self.get_completion(payload)

            return generated_text

        except Exception as e:
            logging.error(f"Error processing RAG query: {e}")
            return f"Error: {e}"

    def get_completion(self, payload: dict) -> str:
        try:
            # Make the call to Bedrock Agent Runtime
            response = self.bedrock_agent_runtime.generate_text(**payload)

            # Extract the sessionId generated or existing
            self.session_id = response.get('sessionId', self.session_id)

                  # Verificar se a resposta contém o campo "output" e "text"
            if 'output' in response and 'text' in response['output']:
                return response['output']['text']
            else:
                return "Nenhuma resposta gerada ou campo 'text' ausente."

        except Exception as e:
            logging.error(f"Error getting response: {e}")
            return f"Error: {e}"
