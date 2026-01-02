"""
Medical AI Agent with RAG Integration and Langfuse Observability
This module implements a CrewAI-based medical assistant agent that interfaces
with a RAG (Retrieval-Augmented Generation) system for medical knowledge.
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional

from rag_tools import get_medical_tools, medical_query_tool, medical_search_tool
from config import get_settings, create_directories
from logger import setup_logging, get_logger

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Get configuration
config = get_settings()

# Initialize Langfuse for observability (if enabled)
langfuse = None
langfuse_handler = None
observe = None

if config.LANGFUSE_ENABLED:
    try:
        from langfuse import observe as langfuse_observe, get_client, propagate_attributes
        from langfuse.langchain import CallbackHandler
        
        # Initialize Langfuse client
        langfuse = get_client()
        
        # Create Langfuse callback handler for LangChain
        langfuse_handler = CallbackHandler(
            public_key=config.LANGFUSE_PUBLIC_KEY        )
        
        observe = langfuse_observe
        logger.info("Langfuse observability enabled")
    except ImportError as e:
        logger.warning(f"Langfuse not available: {e}. Continuing without observability.")
    except Exception as e:
        logger.warning(f"Failed to initialize Langfuse: {e}. Continuing without observability.")

# Create a no-op decorator and context manager if Langfuse is not available
if observe is None:
    def observe(name=None, **kwargs):
        def decorator(func):
            return func
        return decorator

# Create no-op propagate_attributes if Langfuse not available
try:
    from langfuse import propagate_attributes
except ImportError:
    from contextlib import contextmanager
    @contextmanager
    def propagate_attributes(**kwargs):
        yield

# Initialize Google Gemini LLM with Langfuse callback if available
callbacks = [langfuse_handler] if langfuse_handler else []
gemini_llm = ChatGoogleGenerativeAI(
    model=config.AGENT_MODEL,
    google_api_key=config.GOOGLE_API_KEY,
    temperature=0.7,
    convert_system_message_to_human=True,
    callbacks=callbacks
)

@observe(name="create_medical_assistant_agent", capture_input=False, capture_output=False)
def create_medical_assistant_agent() -> Agent:
    """
    Create the primary Medical Assistant agent.
    This agent handles medical queries by leveraging the RAG system.
    """
    logger.info("Creating Medical Assistant Agent...")
    
    medical_assistant = Agent(
        role='Senior Medical Information Specialist',
        goal='Provide accurate, evidence-based medical information to users by querying the medical knowledge base',
        backstory="""You are an expert medical information specialist with extensive experience 
        in healthcare communication. You have access to a comprehensive medical knowledge base 
        containing medical Q&A datasets, doctor-patient conversations, and health information.
        
        Your role is to:
        1. Understand patient questions and concerns
        2. Search the medical knowledge base for relevant information
        3. Synthesize information from multiple sources
        4. Provide clear, accurate, and compassionate responses
        5. Always recommend consulting healthcare professionals for medical decisions
        
        You never provide specific medical advice or diagnoses - you only share educational
        medical information from trusted sources.""",
        verbose=config.AGENT_VERBOSE,
        allow_delegation=False,
        llm=gemini_llm,
        tools=get_medical_tools(),
        max_iter=config.AGENT_MAX_ITERATIONS
    )
    
    logger.info("Medical Assistant Agent created successfully")
    return medical_assistant


@observe(name="create_medical_researcher_agent", capture_input=False, capture_output=False)
def create_medical_researcher_agent() -> Agent:
    """
    Create a Medical Research agent for in-depth analysis.
    This agent performs deeper research on medical topics.
    """
    logger.info("Creating Medical Researcher Agent...")
    
    researcher = Agent(
        role='Medical Research Analyst',
        goal='Conduct thorough research on medical topics using the knowledge base and provide comprehensive analysis',
        backstory="""You are a skilled medical research analyst specializing in 
        synthesizing medical literature and evidence. You excel at:
        
        1. Searching multiple sources for relevant medical information
        2. Analyzing and comparing different medical perspectives
        3. Identifying key findings and evidence
        4. Presenting complex medical information in an accessible way
        5. Highlighting important considerations and limitations
        
        You use the medical search tool to gather comprehensive information
        before forming your analysis.""",
        verbose=config.AGENT_VERBOSE,
        allow_delegation=False,
        llm=gemini_llm,
        tools=[medical_search_tool],
        max_iter=config.AGENT_MAX_ITERATIONS
    )
    
    logger.info("Medical Researcher Agent created successfully")
    return researcher


def create_medical_educator_agent() -> Agent:
    """
    Create a Medical Education agent for explaining medical concepts.
    This agent specializes in making medical information understandable.
    """
    logger.info("Creating Medical Educator Agent...")
    
    educator = Agent(
        role='Medical Health Educator',
        goal='Explain medical concepts in simple, understandable terms while ensuring accuracy',
        backstory="""You are an experienced health educator with a talent for explaining
        complex medical concepts in simple, clear language. Your expertise includes:
        
        1. Breaking down medical terminology into everyday language
        2. Using analogies and examples to explain concepts
        3. Ensuring information is accurate while being accessible
        4. Addressing common misconceptions
        5. Empowering patients with knowledge while emphasizing professional consultation
        
        You take information from the knowledge base and transform it into
        patient-friendly educational content.""",
        verbose=config.AGENT_VERBOSE,
        allow_delegation=False,
        llm=gemini_llm,
        tools=[medical_query_tool],
        max_iter=config.AGENT_MAX_ITERATIONS
    )
    
    logger.info("Medical Educator Agent created successfully")
    return educator


def create_medical_query_task(agent: Agent, question: str) -> Task:
    """Create a task for answering a medical question"""
    return Task(
        description=f"""Answer the following medical question using the medical knowledge base:

Question: {question}

Instructions:
1. Use the medical_knowledge_query tool to search for relevant information
2. Analyze the retrieved information and source documents
3. Provide a clear, accurate answer based on the evidence
4. Include relevant details from the sources
5. Add a disclaimer about consulting healthcare professionals

Ensure your response is:
- Accurate and evidence-based
- Clear and easy to understand
- Properly referenced to source materials
- Includes appropriate medical disclaimers""",
        agent=agent,
        expected_output="""A comprehensive answer to the medical question that includes:
1. A clear response to the question
2. Supporting information from the knowledge base
3. Source references where applicable
4. A reminder to consult healthcare professionals"""
    )


def create_research_task(agent: Agent, topic: str) -> Task:
    """Create a task for researching a medical topic"""
    return Task(
        description=f"""Conduct comprehensive research on the following medical topic:

Topic: {topic}

Instructions:
1. Use the medical_document_search tool to find relevant documents
2. Search for different aspects of the topic (causes, symptoms, treatments, etc.)
3. Compile and synthesize the information
4. Identify key findings and evidence
5. Note any important considerations or limitations

Provide a thorough research report covering:
- Overview of the topic
- Key findings from the knowledge base
- Important medical considerations
- Recommendations for further reading or professional consultation""",
        agent=agent,
        expected_output="""A comprehensive research report including:
1. Topic overview
2. Key findings with source references
3. Analysis of different aspects
4. Important considerations and limitations
5. Recommendations"""
    )


def create_education_task(agent: Agent, concept: str) -> Task:
    """Create a task for explaining a medical concept"""
    return Task(
        description=f"""Explain the following medical concept in simple, understandable terms:

Concept: {concept}

Instructions:
1. Use the medical knowledge base to gather accurate information
2. Break down complex terminology into everyday language
3. Use analogies or examples where helpful
4. Address common questions or misconceptions
5. Keep the explanation accessible but accurate

Your explanation should be suitable for a patient or general audience
without medical training.""",
        agent=agent,
        expected_output="""A clear, patient-friendly explanation that includes:
1. Simple definition of the concept
2. Easy-to-understand explanation
3. Helpful analogies or examples
4. Common questions answered
5. When to seek professional help"""
    )


@observe(name="run_medical_assistant")
def run_medical_assistant(question: str, session_id: str = None) -> str:
    """
    Run the medical assistant to answer a single question.
    
    Args:
        question: The medical question to answer
        session_id: Optional session ID for tracking
    
    Returns:
        The assistant's response
    """
    logger.info(f"Processing medical question: {question[:100]}...")
    
    # Use propagate_attributes to set session_id and metadata for the trace
    with propagate_attributes(
        session_id=session_id,
        metadata={
            "framework": "CrewAI",
            "llm": config.AGENT_MODEL,
            "operation": "medical_query"
        },
        tags=["medical-assistant", "rag-query"],
        trace_name="Medical Assistant Query"
    ):
        try:
            # Update trace with input
            if langfuse:
                langfuse.update_current_trace(
                    input={"question": question}
                )
            
            # Create agent and task
            assistant = create_medical_assistant_agent()
            task = create_medical_query_task(assistant, question)
            
            # Create and run crew
            crew = Crew(
                agents=[assistant],
                tasks=[task],
                process=Process.sequential,
                verbose=config.AGENT_VERBOSE,
                memory=False
            )
            
            result = crew.kickoff()
            result_str = str(result)
            
            logger.info("Medical assistant completed successfully")
            
            # Update trace with output
            if langfuse:
                langfuse.update_current_trace(
                    output={
                        "response": result_str[:500] + "..." if len(result_str) > 500 else result_str,
                        "full_length": len(result_str)
                    }
                )
                
                # Score the trace
                langfuse.score_current_trace(
                    name="output_quality",
                    value=1.0 if len(result_str) > 100 else 0.5,
                    data_type="NUMERIC",
                    comment="Automated quality check based on output length"
                )
            
            return result_str
            
        except Exception as e:
            logger.error(f"Error in medical assistant: {e}")
            # Log error to Langfuse
            if langfuse:
                langfuse.update_current_trace(
                    metadata={"error": str(e)}
                )
                langfuse.score_current_trace(
                    name="execution_status",
                    value=0.0,
                    data_type="NUMERIC",
                    comment=f"Execution failed: {str(e)}"
                )
            raise


@observe(name="run_medical_research")
def run_medical_research(topic: str, session_id: str = None) -> str:
    """
    Run a comprehensive medical research on a topic.
    
    Args:
        topic: The medical topic to research
        session_id: Optional session ID for tracking
    
    Returns:
        The research report
    """
    logger.info(f"Starting medical research on: {topic}")
    
    # Use propagate_attributes to set session_id and metadata for the trace
    with propagate_attributes(
        session_id=session_id,
        metadata={
            "framework": "CrewAI",
            "llm": config.AGENT_MODEL,
            "operation": "medical_research"
        },
        tags=["medical-research", "rag-search", "multi-agent"],
        trace_name="Medical Research Analysis"
    ):
        try:
            # Update trace with input
            if langfuse:
                langfuse.update_current_trace(
                    input={"topic": topic}
                )
            
            # Create agents
            researcher = create_medical_researcher_agent()
            educator = create_medical_educator_agent()
            
            # Create tasks
            research_task = create_research_task(researcher, topic)
            education_task = create_education_task(educator, topic)
            
            # Create and run crew with both agents
            crew = Crew(
                agents=[researcher, educator],
                tasks=[research_task, education_task],
                process=Process.sequential,
                verbose=config.AGENT_VERBOSE,
                memory=False
            )
            
            result = crew.kickoff()
            result_str = str(result)
            
            logger.info("Medical research completed successfully")
            
            # Update trace with output
            if langfuse:
                langfuse.update_current_trace(
                    output={
                        "response": result_str[:500] + "..." if len(result_str) > 500 else result_str,
                        "full_length": len(result_str)
                    }
                )
                
                # Score the trace
                langfuse.score_current_trace(
                    name="output_quality",
                    value=1.0 if len(result_str) > 200 else 0.5,
                    data_type="NUMERIC",
                    comment="Automated quality check based on output length"
                )
            
            return result_str
            
        except Exception as e:
            logger.error(f"Error in medical research: {e}")
            # Log error to Langfuse
            if langfuse:
                langfuse.update_current_trace(
                    metadata={"error": str(e)}
                )
                langfuse.score_current_trace(
                    name="execution_status",
                    value=0.0,
                    data_type="NUMERIC",
                    comment=f"Execution failed: {str(e)}"
                )
            raise


def run_interactive_session():
    """
    Run an interactive session with the medical assistant.
    Users can ask questions until they type 'quit' or 'exit'.
    """
    print("\n" + "=" * 60)
    print("🏥 MEDICAL AI ASSISTANT - Interactive Session")
    print("=" * 60)
    print("\nWelcome! I'm your Medical AI Assistant powered by RAG.")
    print("I can help answer medical questions using a comprehensive")
    print("knowledge base of medical Q&A, dialogues, and health information.")
    print("\nCommands:")
    print("  - Type your medical question to get an answer")
    print("  - Type 'research <topic>' for in-depth research")
    print("  - Type 'quit' or 'exit' to end the session")
    print("\n⚠️  DISCLAIMER: This is for educational purposes only.")
    print("Always consult healthcare professionals for medical advice.")
    print("=" * 60 + "\n")
    
    session_count = 0
    
    while True:
        try:
            user_input = input("\n🩺 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using the Medical AI Assistant. Stay healthy!")
                break
            
            session_count += 1
            
            if user_input.lower().startswith('research '):
                topic = user_input[9:].strip()
                print(f"\n📚 Researching: {topic}")
                print("-" * 40)
                result = run_medical_research(topic)
            else:
                print(f"\n🔍 Searching medical knowledge base...")
                print("-" * 40)
                result = run_medical_assistant(user_input)
            
            print(f"\n🤖 Assistant:\n{result}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again or rephrase your question.")


def main():
    """Main entry point"""
    import sys
    
    print("\n" + "🏥 " * 30)
    print("MEDICAL AI AGENT WITH RAG")
    print("🏥 " * 30 + "\n")
    
    # Create necessary directories
    create_directories()
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--interactive" or sys.argv[1] == "-i":
            run_interactive_session()
        elif sys.argv[1] == "--question" or sys.argv[1] == "-q":
            if len(sys.argv) > 2:
                question = " ".join(sys.argv[2:])
                result = run_medical_assistant(question)
                print(f"\n{result}")
            else:
                print("Please provide a question after --question")
        elif sys.argv[1] == "--research" or sys.argv[1] == "-r":
            if len(sys.argv) > 2:
                topic = " ".join(sys.argv[2:])
                result = run_medical_research(topic)
                print(f"\n{result}")
            else:
                print("Please provide a topic after --research")
        else:
            # Treat the argument as a question
            question = " ".join(sys.argv[1:])
            result = run_medical_assistant(question)
            print(f"\n{result}")
    else:
        # Default: run interactive session
        run_interactive_session()


def get_langfuse_stats():
    """Display Langfuse observability status."""
    print("\n" + "=" * 60)
    if config.LANGFUSE_ENABLED and langfuse:
        print("✓ Langfuse Observability ACTIVE")
        print("=" * 60)
        print(f"Dashboard: {config.LANGFUSE_HOST}")
        print("Features Enabled:")
        print("  ✓ Trace Logging")
        print("  ✓ Token Usage Tracking")
        print("  ✓ Performance Metrics")
        print("  ✓ Session Management")
        print("  ✓ Error Tracking")
        print("  ✓ Quality Scoring")
    else:
        print("✗ Langfuse Observability DISABLED")
        print("=" * 60)
        print("Set LANGFUSE_ENABLED=true in .env to enable observability")
    print("=" * 60 + "\n")


def flush_langfuse():
    """Flush all pending Langfuse data."""
    if langfuse_handler:
        langfuse_handler.flush()
    if langfuse:
        langfuse.flush()
        logger.info("Langfuse data flushed successfully")


if __name__ == "__main__":
    # Display observability info
    get_langfuse_stats()
    
    main()
    
    # Flush Langfuse to ensure all data is sent
    flush_langfuse()
    
    if config.LANGFUSE_ENABLED and langfuse:
        print(f"\n✓ All traces have been sent to Langfuse!")
        print(f"✓ View your dashboard at: {config.LANGFUSE_HOST}\n")
