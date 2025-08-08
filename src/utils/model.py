from langchain_core.messages import AIMessage
from langchain_google_vertexai import ChatVertexAI

from src.config.llm_config import get_llm_config
from src.utils.logger import get_logger

RESPONSE_MATCHING_PATTERN = r"Prompt:.*?Output:(.*)"


logger = get_logger(__name__)


def get_model_response(prompt: str) -> dict:
    """Fetch response from the LLM based on the given prompt."""
    config = get_llm_config()
    model_config, generation_config = config["model"], config["generation"]

    try:
        if model_config.get("model_name"):
            response = get_model_response_with_model_name(
                prompt, model_config, generation_config
            )
        else:
            logger.error("No model_name provided in config")
            return {
                "content": "Error: No model_name in config.",
                "response_metadata": {},
            }

        logger.info(f"Prompt: {prompt}\nResponse: {response}")

        if isinstance(response, AIMessage) and hasattr(response, "content"):
            return {
                "content": (
                    response.content if response.content else "No response from model."
                ),
                "response_metadata": (
                    response.response_metadata
                    if hasattr(response, "response_metadata")
                    else {}
                ),
            }

        return {
            "content": str(response) if response else "No response received.",
            "response_metadata": {},
        }

    except Exception as e:
        logger.error(f"Error processing LLM request: {str(e)}")
        return {"content": "Error: Unable to process request.", "response_metadata": {}}


def get_model_response_with_model_name(
    prompt: str, model_config: dict, generation_config: dict
) -> AIMessage:
    """Get model response using direct model name access."""
    model = ChatVertexAI(
        model=model_config["model_name"],
        temperature=generation_config.get("temperature", 0.2),
        project=model_config.get("project_id"),
        location=model_config.get("location"),
    )
    response = model.invoke(prompt)

    logger.info(f"Response from Model: {response}")

    if isinstance(response, AIMessage):
        return response
    return AIMessage(content=str(response))


def get_vertex_model() -> ChatVertexAI:

    config = get_llm_config()
    model_config, generation_config = config["model"], config["generation"]

    return ChatVertexAI(
        model=model_config["model_name"],
        temperature=generation_config.get("temperature", 0.2),
        project=model_config.get("project_id"),
        location=model_config.get("location"),
    )
