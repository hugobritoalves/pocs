import os
import requests
import uuid
from typing import List, Union, Dict, Any, Optional # Adicionado Optional
from pydantic import BaseModel, Field, ValidationError # Adicionado Field e ValidationError
import json
from datetime import datetime

# Tenta importar a função utilitária; define uma substituta se falhar.
try:
    from utils.pipelines.main import pop_system_message
except ImportError:
    # Fallback se a importação falhar
    def pop_system_message(messages: List[Dict[str, Any]]) -> (Optional[Dict[str, Any]], List[Dict[str, Any]]):
        """Fallback: Removes and returns the first message if it's a system message."""
        if messages and messages[0].get("role") == "system":
            return messages[0], messages[1:]
        return None, messages

# --- Template Padrão (Instruções Base) ---
DEFAULT_PROMPT_TEMPLATE_ANIMA = """Você é um assistente que responde perguntas baseando-se estritamente em um contexto fornecido internamente pela busca em uma base de conhecimento. Use apenas as informações recuperadas para formular sua resposta. Seja conciso e direto. Se a informação necessária para responder à pergunta não estiver no contexto recuperado, informe explicitamente que não encontrou a informação na base de conhecimento. Responda sempre em português brasileiro."""
# ------------------------------------------

class Pipeline:
    class Valves(BaseModel):
        ANIMA_API_KEY: str = Field(default="ANIMA_IA")
        ANIMA_MODEL_NAME: str = Field(default="weduka")
        ID_KNOWLEDGE_BASE: str = Field(default="ITHFWZRFDI")
        ANIMA_API_TIMEOUT: int = Field(default=60, gt=0)
        ID_BEDROCK_MODEL: str = Field(default="amazon.nova-lite-v1:0")
        PROMPT_TEMPLATE_ANIMA: str = Field(default=DEFAULT_PROMPT_TEMPLATE_ANIMA)
        ANIMA_API_BASE_URL: str = Field(description="") 

    def __init__(self):
        self.name = "observatorio2" # Nome atualizado ----------------------------------------------------------------------------------
        try:
            anima_api_base_url_env = os.getenv("ANIMA_URL_1")
            if not anima_api_base_url_env:
                raise ValueError("A variável de ambiente ANIMA_URL_1 (ANIMA_API_BASE_URL) não está definida.")

            env_vars = {
                key: os.getenv(key.upper())
                for key in self.Valves.model_fields.keys()
            }
            filtered_vars = {k: v for k, v in env_vars.items() if v is not None}

            filtered_vars["ANIMA_API_BASE_URL"] = anima_api_base_url_env

            if "ANIMA_API_TIMEOUT" in filtered_vars:
                try:
                    filtered_vars["ANIMA_API_TIMEOUT"] = int(filtered_vars["ANIMA_API_TIMEOUT"])
                except ValueError:
                    del filtered_vars["ANIMA_API_TIMEOUT"]

            self.valves = self.Valves(**filtered_vars)
        except ValidationError as e:
            raise RuntimeError(f"Failed to initialize pipeline valves due to validation error: {e}") from e
        except ValueError as e:
            raise RuntimeError(f"Failed to initialize pipeline: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize pipeline: {e}") from e

    def _format_history_string(self, history_messages: List[Dict[str, Any]]) -> str:
        """Formata o histórico de mensagens em uma string."""
        history_str = ""
        for msg in history_messages:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            content_text = ""

            if isinstance(content, str):
                content_text = content
            elif isinstance(content, list):
                text_parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
                content_text = "\n".join(filter(None, text_parts))

            if content_text:
                history_str += f"{role}: {content_text.strip()}\n"

        return history_str.strip()

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> str:
        if not user_message:
            return "Erro: Nenhuma mensagem de usuário fornecida."

        req_user_id = None
        user_info = body.get("user")
        if isinstance(user_info, dict):
                req_user_id = user_info.get("name")
        if not req_user_id and body.get("user_id"):
                req_user_id = body.get("user_id")

        if not req_user_id:
                return "Erro: ID do usuário ('user.id', 'user.name', or 'user_id') não encontrado na requisição."

        session_id_to_send = ""

        try:
            _system_msg_dict, conversation_messages = pop_system_message(messages)

            if conversation_messages and conversation_messages[-1].get("role") == "user":
                last_msg_content = conversation_messages[-1].get("content")
                is_duplicate = False
                if isinstance(last_msg_content, str) and last_msg_content == user_message:
                    is_duplicate = True
                elif isinstance(last_msg_content, list):
                    text_parts = [item.get("text", "") for item in last_msg_content if isinstance(item, dict) and item.get("type") == "text"]
                    if len(text_parts) == 1 and text_parts[0] == user_message:
                            is_duplicate = True
                if is_duplicate:
                    conversation_messages = conversation_messages[:-1]

            history_string = self._format_history_string(conversation_messages)

            custom_template_instructions = body.get("prompt_template")
            if custom_template_instructions and isinstance(custom_template_instructions, str):
                base_instructions = custom_template_instructions
            else:
                base_instructions = self.valves.PROMPT_TEMPLATE_ANIMA

            prompt_parts = [base_instructions]
            if history_string:
                prompt_parts.append("\n\n--- Histórico da Conversa Anterior ---")
                prompt_parts.append(history_string)
                prompt_parts.append("--- Fim do Histórico ---")

            prompt_parts.append("\n\n--- Pergunta Atual do Usuário ---")
            prompt_parts.append(user_message)

            prompt_to_send = "\n".join(prompt_parts).strip()

            api_url = f"{self.valves.ANIMA_API_BASE_URL}/{self.valves.ID_KNOWLEDGE_BASE}/retrieveandgenerate"
            effective_model_id = self.valves.ID_BEDROCK_MODEL

            payload = {
                "userId": str(req_user_id),
                "sessionId": session_id_to_send,
                "prompt": prompt_to_send,
                "modelId": effective_model_id,
                "model": self.valves.ANIMA_MODEL_NAME,
                "apiKey": self.valves.ANIMA_API_KEY,
                "tag": self.valves.ID_KNOWLEDGE_BASE
            }

            final_text_response = self._call_anima_api(api_url, payload)
            return final_text_response

        except Exception as e:
            return f"Ocorreu um erro inesperado no pipeline: {e}"

    def _format_citations(self, citations_data: List[Any]) -> str:
        """Formata as citações recebidas da API."""
        if not citations_data or not isinstance(citations_data, list):
            return ""

        unique_sources = set()
        formatted_references = []

        for citation in citations_data:
                if isinstance(citation, dict):
                    references = citation.get("retrievedReferences", [])
                    for ref in references:
                        location = ref.get("location", {})
                        s3_uri = location.get("s3Location", {}).get("uri")
                        source_identifier = s3_uri
                        if source_identifier and source_identifier not in unique_sources:
                            unique_sources.add(source_identifier)
                            display_source = source_identifier.split('/')[-1] if 's3://' in source_identifier else source_identifier
                            formatted_references.append(display_source)
                elif isinstance(citation, str) and citation:
                    if citation not in unique_sources:
                        unique_sources.add(citation)
                        display_source = citation.split('/')[-1] if '/' in citation else citation
                        formatted_references.append(display_source)

        if not formatted_references:
            return ""

        # Adiciona título e formatação
        references_text = "\n\n**Fontes:**\n"
        for i, source in enumerate(sorted(list(formatted_references))):
            references_text += f"{i+1}. {source}\n"
        # Retorna a string completa, incluindo o título e newlines iniciais
        return references_text.rstrip() # Remove apenas o último newline

    # Função _call_anima_api com a correção aplicada
    def _call_anima_api(self, url: str, payload: Dict[str, Any]) -> str:
        """
        Chama a API Anima LLM, tratando possíveis erros.
        Retorna a string de texto da resposta final (outputText + citações).
        """
        final_response_text = "Erro: Não foi possível obter uma resposta da API." # Default error
        try:
            response = requests.post(
                url,
                json=payload,
                headers={'Content-Type': 'application/json', 'Accept': 'application/json'},
                timeout=self.valves.ANIMA_API_TIMEOUT
            )
            response.raise_for_status() # Verifica erros HTTP 4xx/5xx

            data = response.json()

            # --- CORREÇÃO APLICADA ---
            output_text = data.get('outputText', '').strip()
            # -------------------------

            citations_list = data.get("citations", [])

            if not isinstance(citations_list, list):
                    citations_list = []

            citations_text = self._format_citations(citations_list) # Retorna string com \n\n**Fontes:**... ou ""

            # --- LÓGICA DE COMBINAÇÃO AJUSTADA ---
            if output_text:
                final_response_text = output_text
                if citations_text:
                    # Adiciona as citações formatadas (que já incluem o título e espaçamento)
                    final_response_text += citations_text
            elif citations_text:
                # Se NÃO houver texto principal, mas houver citações, retorna apenas as citações formatadas.
                # Remove o espaço inicial adicionado por _format_citations se for a única coisa retornada.
                final_response_text = citations_text.lstrip()
            else:
                # Se não houver nem texto nem citações
                final_response_text = "Não foi possível gerar uma resposta ou encontrar informações relevantes."
            # ------------------------------------

        except requests.exceptions.Timeout:
            final_response_text = f"Erro: A API demorou muito para responder (timeout de {self.valves.ANIMA_API_TIMEOUT}s)."
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            error_detail = f"Erro da API Anima (Status: {status_code})"
            if status_code == 429:
                    final_response_text = "Erro: Muitas requisições para a API. Tente novamente mais tarde."
            elif 400 <= status_code < 500:
                    final_response_text = f"Erro na requisição para a API ({status_code})."
            else: # 5xx
                    final_response_text = f"Erro interno na API Anima ({status_code}). Tente novamente mais tarde."
        except requests.exceptions.ConnectionError as e:
                final_response_text = f"Erro de conexão ao tentar acessar a API Anima."
        except requests.exceptions.RequestException as e:
                final_response_text = f"Erro de comunicação com a API Anima: {e}"
        except json.JSONDecodeError as e:
            final_response_text = "Erro: Resposta inválida (não JSON) da API Anima."
        except Exception as e:
            final_response_text = f"Ocorreu um erro inesperado na chamada da API: {e}"

        return final_response_text