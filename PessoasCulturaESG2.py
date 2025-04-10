import os
import json
from datetime import datetime
from typing import List, Union, Dict, Any, Optional, Tuple

import requests
from pydantic import BaseModel, Field, ValidationError

# --- Fallback Function ---
try:
    from utils.pipelines.main import pop_system_message
except ImportError:
    print("WARNING: Could not import 'pop_system_message' from 'utils.pipelines.main'. Using fallback.")
    def pop_system_message(messages: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """Fallback: Removes and returns the first message if it's a system message."""
        if messages and messages[0].get("role") == "system":
            return messages[0], messages[1:]
        return None, messages

DEFAULT_PROMPT_TEMPLATE_ANIMA = """Você é um assistente de IA. Sua função principal é responder perguntas usando APENAS informações recuperadas de uma base de conhecimento interna. Siga estas regras de conversa RIGOROSAMENTE:

**ETAPA 1: VERIFICAR INFORMAÇÕES DO USUÁRIO**
Antes de qualquer outra coisa, analise o histórico da conversa. O usuário já informou claramente qual é o seu **Tipo de Vínculo** E qual é a sua **IES (Instituição de Ensino Superior)**?

**ETAPA 2: AÇÃO CONDICIONAL**

* **CASO A: FALTA VÍNCULO OU IES (ou ambos)**
    * Se o Tipo de Vínculo ou a IES (ou ambos) NUNCA foram mencionados antes na conversa:
    * **NÃO RESPONDA a pergunta que o usuário acabou de fazer.** Ignore-a por enquanto.
    * Sua ÚNICA resposta DEVE SER fazer as seguintes perguntas, incluindo as opções como exemplo:
        "Para que eu possa te ajudar melhor, por favor, me informe:
        a) Qual o seu tipo de vínculo? (Ex: Administrativo, Docente, Bolsista, Estagiário, Jovem Aprendiz, Autônomo)
        b) Qual a sua IES (Instituição de Ensino Superior)? (Ex: Ages, FPB, Una, Anhembi Morumbi, Gama Academy, UniBH, HSM, UniCuritiba, UNIFACS, UniFG (BA/PE), UniRitter, Inspirali, Le Cordon Bleu SP, Instituto Ânima, CEDEPE, Business School SP, A2S Tecnologia, Ânima (Estrutura Corporativa))"
    * Não adicione mais nada à sua resposta. Apenas essas perguntas.

* **CASO B: VÍNCULO E IES JÁ CONHECIDOS**
    * Se o Tipo de Vínculo E a IES já foram informados pelo usuário em mensagens anteriores desta conversa:
    * Prossiga para responder a pergunta MAIS RECENTE do usuário.
    * Use o Vínculo e a IES como contexto adicional, se relevante.
    * Responda baseando-se ESTRITAMENTE nas informações recuperadas da base de conhecimento fornecida internamente.
    * Seja conciso e direto.
    * Se a informação necessária para responder à pergunta do usuário não estiver no contexto recuperado, informe explicitamente: "Não encontrei a informação sobre isso na base de conhecimento."
    * Responda sempre em português brasileiro.

**NUNCA use conhecimento externo à base fornecida.**
"""

class Pipeline:
    class Valves(BaseModel):
        # Nomes dos campos EXATAMENTE como devem ser referenciados
        ANIMA_API_KEY: str = Field(default="ANIMA_IA")
        ANIMA_MODEL_NAME: str = Field(default="PessoasCulturaESG")
        ID_KNOWLEDGE_BASE: str = Field(default="KSETSXIHN4")
        ID_BEDROCK_MODEL: str = Field(default="amazon.nova-lite-v1:0")
        ANIMA_API_TIMEOUT: int = Field(default=60, gt=0, description="Timeout em segundos para a chamada da API Anima.")
        PROMPT_TEMPLATE_ANIMA: str = Field(default=DEFAULT_PROMPT_TEMPLATE_ANIMA, description="Template base do prompt do sistema.")
        ANIMA_API_BASE_URL: str = Field(description="URL base da API Anima (ex: http://localhost:8000). Definido via env var ANIMA_URL_1.")

    # --- __INIT__ CORRIGIDO ---
    def __init__(self):
        self.name = "Pessoas,Cultura e ESG (2)"
        try:
            # 1. Obter a URL base obrigatória da env var
            anima_api_base_url_env = os.getenv("ANIMA_URL_1")
            if not anima_api_base_url_env:
                raise ValueError("A variável de ambiente ANIMA_URL_1 (para ANIMA_API_BASE_URL) não está definida.")

            # 2. Inicializar dicionário para dados de configuração
            config_data: Dict[str, Any] = {}

            # 3. Iterar sobre os nomes de campo definidos em Valves (são case-sensitive, ex: 'ANIMA_API_KEY')
            for field_name in self.Valves.model_fields.keys():
                # Pula ANIMA_API_BASE_URL, pois já foi lida e será adicionada depois
                if field_name == "ANIMA_API_BASE_URL":
                    continue

                # Busca a variável de ambiente correspondente (geralmente em maiúsculas)
                env_value = os.getenv(field_name.upper())

                if env_value is not None:
                    # Tratamento especial para timeout (conversão para int)
                    if field_name == "ANIMA_API_TIMEOUT":
                        try:
                            # Usa o NOME DO CAMPO (ex: 'ANIMA_API_TIMEOUT') como chave
                            config_data[field_name] = int(env_value)
                        except ValueError:
                            print(f"WARNING: {field_name.upper()} inválido ('{env_value}'). Usando default.")
                            # Não adiciona ao dict, Pydantic usará seu default
                    else:
                        # Usa o NOME DO CAMPO (ex: 'ANIMA_API_KEY') como chave
                        config_data[field_name] = env_value

            # 4. Adiciona a URL base obrigatória usando a chave correta (maiúscula)
            config_data["ANIMA_API_BASE_URL"] = anima_api_base_url_env.strip() # Adiciona strip para remover espaços extras

            # 5. Valida a configuração usando Pydantic (agora as chaves correspondem aos campos)
            self.valves = self.Valves(**config_data)

            # Define constantes após validação bem-sucedida
            self.not_found_indicator = "Não encontrei a informação sobre isso na base de conhecimento"
            self.allowed_citation_extensions = {
                '.txt', '.pdf', '.doc', '.docx', '.ppt', '.pptx',
                '.xlsx', '.xls', '.csv', '.json', '.jpeg',
                '.jpg', '.png', '.md'
            }
            self.max_citation_filename_length = 50

        except ValidationError as e:
            # Melhora a mensagem de erro para Pydantic
            error_details = e.errors()
            field_errors = "; ".join([f"Campo '{err['loc'][0]}': {err['msg']}" for err in error_details if err.get('loc')])
            raise RuntimeError(f"Falha ao inicializar configuração do pipeline (Valves). Erros: {field_errors}") from e
        except ValueError as e:
            raise RuntimeError(f"Falha ao inicializar o pipeline (ValueError): {e}") from e
        except Exception as e:
            # import traceback; traceback.print_exc() # Descomente para debug detalhado
            raise RuntimeError(f"Falha inesperada ao inicializar o pipeline: {type(e).__name__} - {e}") from e
    # --- FIM DO __INIT__ CORRIGIDO ---

    # Funções _format_history_string, pipe, _format_citations, _call_anima_api
    # permanecem iguais às da última versão otimizada e corrigida
    # (Coloque as versões corretas dessas funções aqui)

    def _format_history_string(self, history_messages: List[Dict[str, Any]]) -> str:
        """
        Formata o histórico de mensagens, removendo fontes de msgs anteriores do assistente.
        Otimizado para usar list.append + join.
        """
        history_parts: List[str] = []
        sources_marker = "\n\n**Fontes:**"

        for msg in history_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            content_text = ""

            if isinstance(content, str):
                content_text = content
            elif isinstance(content, list):
                text_parts = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                content_text = "\n".join(filter(None, text_parts))

            if not content_text:
                continue

            if role == "assistant":
                content_text = content_text.split(sources_marker, 1)[0].rstrip()

            if content_text:
                 history_parts.append(f"{role.capitalize()}: {content_text}")

        return "\n".join(history_parts)


    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> str:
        """
        Ponto de entrada principal do pipeline.
        Processa a mensagem, formata histórico/prompt, chama API e retorna resposta.
        """
        if not user_message:
            return "Erro: Nenhuma mensagem de usuário fornecida."

        req_user_id: Optional[str] = None
        user_info: Optional[Dict] = body.get("user")

        if isinstance(user_info, dict):
            req_user_id = user_info.get("name")

        if not req_user_id:
            req_user_id = body.get("user_id")

        if not req_user_id:
            return "Erro: ID do usuário ('user.name' ou 'user_id') não encontrado na requisição."

        session_id_to_send = ""

        try:
            _system_msg_dict, conversation_messages = pop_system_message(messages)

            if conversation_messages and conversation_messages[-1].get("role") == "user":
                last_msg_content = conversation_messages[-1].get("content")
                is_duplicate = (isinstance(last_msg_content, str) and last_msg_content == user_message) or \
                               (isinstance(last_msg_content, list) and
                                len(last_msg_content) == 1 and
                                isinstance(last_msg_content[0], dict) and
                                last_msg_content[0].get("type") == "text" and
                                last_msg_content[0].get("text") == user_message)
                if is_duplicate:
                    conversation_messages = conversation_messages[:-1]

            history_string = self._format_history_string(conversation_messages)

            base_instructions = body.get("prompt_template", self.valves.PROMPT_TEMPLATE_ANIMA)
            if not isinstance(base_instructions, str) or not base_instructions:
                 base_instructions = self.valves.PROMPT_TEMPLATE_ANIMA

            prompt_parts: List[str] = [base_instructions]
            if history_string:
                prompt_parts.extend([
                    "\n\n--- Histórico da Conversa Anterior ---",
                    history_string,
                    "--- Fim do Histórico ---"
                ])

            user_message_with_instruction = f"{user_message.strip()}\n\n(Instrução: Por favor, me responda em português brasileiro.)"
            prompt_parts.extend([
                "\n\n--- Pergunta Atual do Usuário ---",
                user_message_with_instruction
            ])

            prompt_to_send = "\n".join(prompt_parts).strip()

            api_url = f"{self.valves.ANIMA_API_BASE_URL.rstrip('/')}/{self.valves.ID_KNOWLEDGE_BASE}/retrieveandgenerate"

            payload = {
                "userId": str(req_user_id),
                "sessionId": session_id_to_send,
                "prompt": prompt_to_send,
                "modelId": self.valves.ID_BEDROCK_MODEL,
                "model": self.valves.ANIMA_MODEL_NAME,
                "apiKey": self.valves.ANIMA_API_KEY,
                "tag": self.valves.ID_KNOWLEDGE_BASE
            }

            final_text_response = self._call_anima_api(api_url, payload)
            return final_text_response

        except Exception as e:
            print(f"ERROR in pipeline: {e}")
            return f"Ocorreu um erro inesperado ao processar sua solicitação: {type(e).__name__}"


    def _format_citations(self, citations_data: List[Any]) -> str:
        """
        Formata as citações, aplicando filtros de extensão e comprimento.
        Otimizado para usar list.append + join.
        """
        if not citations_data or not isinstance(citations_data, list):
            return ""

        unique_source_identifiers = set()
        valid_references_to_display: List[str] = []

        for citation in citations_data:
            source_identifier: Optional[str] = None
            display_source: Optional[str] = None

            if isinstance(citation, dict):
                references = citation.get("retrievedReferences", [])
                if isinstance(references, list):
                    for ref in references:
                         if isinstance(ref, dict):
                            location = ref.get("location", {})
                            if isinstance(location, dict):
                                s3_uri = location.get("s3Location", {}).get("uri")
                                if s3_uri and isinstance(s3_uri, str):
                                     source_identifier = s3_uri
                                     display_source = s3_uri.split('/')[-1]
                                     break
            elif isinstance(citation, str) and citation:
                 source_identifier = citation
                 display_source = citation.split('/')[-1] if '/' in citation else citation

            if display_source and source_identifier and source_identifier not in unique_source_identifiers:
                 if len(display_source) <= self.max_citation_filename_length:
                     _ , file_extension = os.path.splitext(display_source)
                     if file_extension.lower() in self.allowed_citation_extensions:
                         unique_source_identifiers.add(source_identifier)
                         valid_references_to_display.append(display_source)

        if not valid_references_to_display:
            return ""

        formatted_lines = [
            f"{i}. {source}"
            for i, source in enumerate(sorted(valid_references_to_display), 1)
        ]
        return "\n\n**Fontes:**\n" + "\n".join(formatted_lines)


    def _call_anima_api(self, url: str, payload: Dict[str, Any]) -> str:
        """
        Chama a API Anima LLM com tratamento de erros robusto.
        """
        final_response_text: str
        try:
            response = requests.post(
                url,
                json=payload,
                headers={'Content-Type': 'application/json', 'Accept': 'application/json'},
                timeout=self.valves.ANIMA_API_TIMEOUT
            )
            response.raise_for_status()

            data = response.json()
            output_text = data.get('outputText', '').strip()

            if self.not_found_indicator in output_text:
                final_response_text = output_text
            else:
                citations_list = data.get("citations", [])
                if not isinstance(citations_list, list):
                    citations_list = []

                citations_text = self._format_citations(citations_list)

                if output_text:
                    final_response_text = output_text + citations_text
                elif citations_text:
                    final_response_text = citations_text.lstrip()
                else:
                    final_response_text = "Não foi possível gerar uma resposta com as informações disponíveis."

        except requests.exceptions.Timeout:
            final_response_text = f"Erro: A API em {url} demorou muito para responder (timeout de {self.valves.ANIMA_API_TIMEOUT}s)."
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            error_body_text = ""
            try:
                error_body = e.response.json()
                error_body_text = f" Detalhe: {json.dumps(error_body)}"
            except json.JSONDecodeError:
                error_body_text = f" Detalhe (raw): {e.response.text[:200]}..."

            if status_code == 401 or status_code == 403:
                 final_response_text = f"Erro de Autenticação/Autorização ({status_code}) ao acessar a API. Verifique a API Key."
            elif status_code == 404:
                 final_response_text = f"Erro ({status_code}): Recurso não encontrado em {url}. Verifique ID da Base/Endpoint."
            elif status_code == 429:
                final_response_text = "Erro: Limite de requisições para a API excedido. Tente novamente mais tarde."
            elif 400 <= status_code < 500:
                final_response_text = f"Erro na requisição ({status_code}) para a API. Verifique os dados enviados.{error_body_text}"
            else: # 5xx
                final_response_text = f"Erro interno na API Anima ({status_code}). Tente novamente mais tarde.{error_body_text}"
        except requests.exceptions.ConnectionError:
            final_response_text = f"Erro de conexão ao tentar acessar a API Anima em {url}."
        except requests.exceptions.RequestException as e:
            final_response_text = f"Erro inesperado de comunicação com a API Anima: {e}"
        except json.JSONDecodeError:
             final_response_text = f"Erro: Resposta inválida (não JSON) recebida da API em {url}."
        except Exception as e:
            # import logging; logging.exception("Erro não tratado em _call_anima_api")
            print(f"ERROR in _call_anima_api: {e}")
            final_response_text = f"Ocorreu um erro inesperado durante a chamada da API: {type(e).__name__}"

        return final_response_text.strip()