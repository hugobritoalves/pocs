"""
title: Langfuse Filter Pipeline
author: open-webui
date: 2024-09-27
version: 1.4
license: MIT
description: A filter pipeline that uses Langfuse.
requirements: langfuse
"""

from typing import List, Optional
import os
import uuid

from utils.pipelines.main import get_last_assistant_message
from pydantic import BaseModel
from langfuse import Langfuse
from langfuse.api.resources.commons.errors.unauthorized_error import UnauthorizedError

def get_last_assistant_message_obj(messages: List[dict]) -> dict:
    for message in reversed(messages):
        if message["role"] == "assistant":
            return message
    return {}


class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        secret_key: str
        public_key: str
        host: str

    def __init__(self):
        self.type = "filter"
        self.name = "Langfuse Filter"
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],
                "secret_key": os.getenv("LANGFUSE_SECRET_KEY", "your-secret-key-here"),
                "public_key": os.getenv("LANGFUSE_PUBLIC_KEY", "your-public-key-here"),
                "host": os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            }
        )
        self.langfuse = None
        self.chat_traces = {}
        self.chat_generations = {}

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        self.set_langfuse()

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        if self.langfuse:
            self.langfuse.flush()

    async def on_valves_updated(self):
        self.set_langfuse()

    def set_langfuse(self):
        try:
            self.langfuse = Langfuse(
                secret_key=self.valves.secret_key,
                public_key=self.valves.public_key,
                host=self.valves.host,
                debug=False,
            )
            if self.langfuse:
                self.langfuse.auth_check()
        except UnauthorizedError:
            print("Langfuse credentials incorrect. Please check your credentials.")
        except Exception as e:
            print(f"Langfuse error: {str(e)}")

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"inlet:{__name__}")
        
        # Remover chat_id do payload antes de processar
        langfuse_chat_id = body.pop("chat_id", f"webui-{uuid.uuid4()}")  # ← Correção crítica aqui
        
        # Validação dos campos obrigatórios
        required_keys = ["model", "messages"]
        if missing := [key for key in required_keys if key not in body]:
            raise ValueError(f"Missing keys: {', '.join(missing)}")

        try:
            # Criação do trace com session_id próprio
            trace = self.langfuse.trace(
                name="openwebui-chat",
                input=body,
                user_id=user.get("email", "anonymous"),
                metadata={
                    "user_name": user.get("name", "unknown"),
                    "user_id": user.get("id", "unknown")
                },
                session_id=langfuse_chat_id  # Usa ID gerado/removido
            )

            # Criação da generation
            generation = trace.generation(
                name=body["model"],
                model=body["model"],
                input=body["messages"],
                metadata={
                    "interface": "open-webui",
                    "webui_chat_id": langfuse_chat_id  # Mantém referência interna
                }
            )

            # Armazena referências
            self.chat_traces[langfuse_chat_id] = trace
            self.chat_generations[langfuse_chat_id] = generation

        except Exception as e:
            print(f"Langfuse tracking error: {str(e)}")

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"outlet:{__name__}")
        
        # Extrai o chat_id interno do WebUI
        langfuse_chat_id = f"webui-{body.get('chat_id', 'unknown')}"  # Não existe mais no body
        
        if langfuse_chat_id not in self.chat_generations:
            return body

        try:
            trace = self.chat_traces[langfuse_chat_id]
            generation = self.chat_generations[langfuse_chat_id]
            
            # Processa mensagem do assistente
            assistant_message = get_last_assistant_message(body["messages"])
            assistant_message_obj = get_last_assistant_message_obj(body["messages"])
            
            # Coleta métricas de uso
            usage = None
            if assistant_message_obj and isinstance(assistant_message_obj.get("info"), dict):
                info = assistant_message_obj["info"]
                usage = {
                    "input": info.get("prompt_eval_count") or info.get("prompt_tokens"),
                    "output": info.get("eval_count") or info.get("completion_tokens"),
                    "unit": "TOKENS"
                }

            # Atualiza registros no Langfuse
            trace.update(output=assistant_message)
            generation.end(
                output=assistant_message,
                usage=usage,
                metadata={"status": "completed"}
            )

        except Exception as e:
            print(f"Langfuse update error: {str(e)}")
        finally:
            # Limpa registros
            self.chat_traces.pop(langfuse_chat_id, None)
            self.chat_generations.pop(langfuse_chat_id, None)

        return body