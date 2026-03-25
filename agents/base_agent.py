import json
import time
import requests
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class AgentResult:
    agent_name: str
    status: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metrics: Dict[str, Any] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    narrative: str = ""
    raw_analysis: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    execution_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "status": self.status,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "insights": self.insights,
            "recommendations": self.recommendations,
            "narrative": self.narrative,
            "raw_analysis": self.raw_analysis,
            "error_message": self.error_message,
            "execution_time_seconds": self.execution_time_seconds,
        }


class BaseAgent:
    GEMINI_MODEL = "gemini-2.5-flash"
    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    MAX_RETRIES = 2
    RETRY_DELAY_SECONDS = 3
    MAX_OUTPUT_TOKENS = 8192
    TEMPERATURE = 0.3

    def __init__(self, api_key: str, agent_name: str = "BaseAgent"):
        if not api_key:
            raise ValueError("API Key de Gemini es obligatoria.")
        self._api_key = api_key
        self.agent_name = agent_name
        self._log: List[str] = []

    def _log_event(self, message: str):
        entry = f"[{datetime.now().strftime('%H:%M:%S')}] [{self.agent_name}] {message}"
        self._log.append(entry)

    def get_log(self) -> List[str]:
        return self._log.copy()

    def _call_gemini(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        force_json: bool = True,
    ) -> Dict[str, Any]:
        url = (
            f"{self.GEMINI_BASE_URL}/{self.GEMINI_MODEL}"
            f":generateContent?key={self._api_key}"
        )
        temp = temperature if temperature is not None else self.TEMPERATURE
        tokens = max_tokens if max_tokens is not None else self.MAX_OUTPUT_TOKENS

        generation_config = {
            "temperature": temp,
            "maxOutputTokens": tokens,
        }
        if force_json:
            generation_config["responseMimeType"] = "application/json"

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"{system_prompt}\n\n---\n\n{user_prompt}"}],
                }
            ],
            "generationConfig": generation_config,
        }

        last_error = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                self._log_event(f"Llamando a Gemini (intento {attempt})...")
                response = requests.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=120,
                )

                if response.status_code == 200:
                    data = response.json()
                    text = (
                        data.get("candidates", [{}])[0]
                        .get("content", {})
                        .get("parts", [{}])[0]
                        .get("text", "")
                    )
                    if text:
                        self._log_event("Respuesta recibida exitosamente.")
                        if force_json:
                            try:
                                return json.loads(text)
                            except json.JSONDecodeError:
                                return {"raw_text": text}
                        else:
                            return {"raw_text": text}
                    else:
                        last_error = "Respuesta vacia de Gemini"

                elif response.status_code == 429:
                    last_error = "Rate limit"
                    time.sleep(self.RETRY_DELAY_SECONDS * attempt)

                else:
                    last_error = f"HTTP {response.status_code}: {response.text[:300]}"

            except requests.exceptions.Timeout:
                last_error = "Timeout"
            except requests.exceptions.RequestException as e:
                last_error = f"Error de conexion: {str(e)}"

            if attempt < self.MAX_RETRIES:
                time.sleep(self.RETRY_DELAY_SECONDS)

        raise RuntimeError(f"[{self.agent_name}] Fallo tras {self.MAX_RETRIES} intentos: {last_error}")

    def _safe_result(self, error_msg: str, exec_time: float = 0.0) -> AgentResult:
        self._log_event(f"ERROR: {error_msg}")
        return AgentResult(
            agent_name=self.agent_name,
            status="error",
            error_message=error_msg,
            execution_time_seconds=exec_time,
        )

    def analyze(self, df, **kwargs) -> AgentResult:
        raise NotImplementedError("Cada agente debe implementar analyze()")
