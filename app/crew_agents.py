"""
Agentic RAG Module - Crew.AI with Ollama via ChatOpenAI.

Two core agents: 
1. FactValidator (validates facts against context).
2. ResponseFormatter (formats the final response with proper citations).
"""
import logging

from crewai import LLM, Agent, Crew, Task

from app.config import GENERATION_MODEL, OLLAMA_HOST

# Setup logging
logging.basicConfig(level=logging.INFO)
agent_logger = logging.getLogger(__name__)


def initialize_llm_client(model_name: str):
    """Initializes the LLM for CrewAI using the CrewAI LLM wrapper pointing to Ollama."""
    return LLM(
        model=f"ollama/{model_name or GENERATION_MODEL}",
        base_url=OLLAMA_HOST
    )


def create_fact_validator_agent(llm_client):
    """Agent responsible for validating facts using retrieved documents."""
    return Agent(
        role="Fact Validator",
        goal="Validate and summarize core information from retrieved context documents",
        backstory="You meticulously analyze documents to verify facts, ensure accuracy, and cross-check claims against the provided sources.",
        verbose=True,
        allow_delegation=False,
        llm=llm_client
    )


def create_response_formatter_agent(llm_client):
    """Agent responsible for synthesizing and formatting the final response with citations."""
    return Agent(
        role="Response Formatter",
        goal="Create a well-formatted, comprehensive answer with proper citations",
        backstory="You synthesize validated facts into coherent, high-quality answers, adding precise citations [1][2][3] to trace all claims back to their sources.",
        verbose=True,
        allow_delegation=False,
        llm=llm_client
    )


def structure_retrieved_context(context_nodes):
    """Formats retrieved document nodes into a readable string for the agent context, including metadata."""
    if not context_nodes:
        return "No documents available for context."

    formatted_context = []
    for i, node_data in enumerate(context_nodes, 1):
        text_content = node_data.get("text", "")
        relevance_score = node_data.get("score", 0.0)
        # Use document_id or a fallback
        document_identifier = node_data.get("document_id", node_data.get("metadata", {}).get("id", i))
        source_file_name = node_data.get("source", node_data.get("metadata", {}).get("source_file", "unknown"))

        formatted_context.append(
            f"Context Document {i} [ID: {document_identifier}, Source: {source_file_name}, Relevance: {relevance_score:.2f}]:\n{text_content}"
        )

    return "\n\n" + "=" * 50 + "\n\n".join(formatted_context)


def process_with_agents(model_name: str, user_input_query: str, context_documents: list):
    """Processes the user query using the Fact Validator and Response Formatter agents (CrewAI pipeline)."""
    agent_logger.info(f"[AGENT] Starting agent processing for query: {user_input_query[:50]}...")

    if not context_documents:
        agent_logger.warning("[AGENT] No documents retrieved")
        return {
            'response': 'No relevant information found in the knowledge base.',
            'query': user_input_query,
            'sources': [],
            'count': 0
        }

    try:
        agent_logger.info(f"[AGENT] Retrieved {len(context_documents)} documents")
        agent_logger.info("[AGENT] Step 1: Initializing LLM client...")
        llm_client = initialize_llm_client(model_name)

        agent_logger.info("[AGENT] Step 2: Formatting context from retrieved documents...")
        context_string = structure_retrieved_context(context_documents)

        agent_logger.info("[AGENT] Step 3: Creating agents...")
        fact_validator = create_fact_validator_agent(llm_client)
        response_formatter = create_response_formatter_agent(llm_client)

        agent_logger.info("[AGENT] Step 4: Creating tasks...")
        validation_task = Task(
            description=f"Validate and summarize facts required to answer: '{user_input_query}'\n\nContext Documents:\n{context_string}",
            agent=fact_validator,
            expected_output="A concise, validated facts summary that directly supports answering the user query."
        )

        # Extract unique source file names for the formatter context
        source_file_references = []
        seen_sources = set()
        for i, doc in enumerate(context_documents, 1):
            source = doc.get('source', doc.get('metadata', {}).get('source_file', 'unknown'))
            if source != 'unknown' and source not in seen_sources:
                source_file_references.append(f"[{i}] Source: {source}")
                seen_sources.add(source)
        sources_info = "\n".join(source_file_references) if source_file_references else "No sources available"

        formatting_task = Task(
            description=f"""Create the final, comprehensive answer for the user query: '{user_input_query}'.

Instructions:
1. Use the validated facts provided by the Fact Validator.
2. Structure the answer clearly and professionally.
3. Incorporate citations (e.g., [1][2]) for every piece of factual information, referencing the source numbers listed below.
4. At the end of your answer, explicitly mention which source files you referenced.

Available Source Files and References:
{sources_info}""",
            agent=response_formatter,
            expected_output="Complete, coherent answer with integrated citations (e.g., [1][2]) and a concluding list of referenced source file names.",
            context=[validation_task]
        )

        agent_logger.info("[AGENT] Step 5: Initializing and running Crew pipeline...")
        agent_crew = Crew(
            agents=[fact_validator, response_formatter],
            tasks=[validation_task, formatting_task],
            verbose=True
        )

        agent_logger.info("[AGENT] Fact Validator agent: Validating facts...")
        final_response_output = agent_crew.kickoff()
        agent_logger.info("[AGENT] Response Formatter agent: Formatting final output...")
        agent_logger.info("[AGENT] Agent processing completed successfully")

        # Format response metadata and source references
        source_metadata = [
            {
                'id': str(d.get('document_id', i)),
                'text': d['text'][:200],
                'score': d['score'],
                'source': d.get('source', 'unknown'),
                'metadata': d.get('metadata', {})
            }
            for i, d in enumerate(context_documents[:3], 1)
        ]

        return {
            'response': str(final_response_output),
            'query': user_input_query,
            'sources': source_metadata,
            'count': len(source_metadata)
        }

    except Exception as e:
        agent_logger.error(f"[AGENT] Error during processing: {e}", exc_info=True)

        # Fallback response (using initial retrieved text)
        fallback_text = f"An error occurred during agent processing. Based on the initial context retrieved: {context_documents[0]['text'][:300]}..."
        source_metadata = [
            {'id': str(i), 'text': d['text'][:200], 'score': d['score']}
            for i, d in enumerate(context_documents[:3], 1)
        ]

        return {
            'response': fallback_text,
            'query': user_input_query,
            'sources': source_metadata,
            'count': len(source_metadata),
            'error': str(e)
        }
