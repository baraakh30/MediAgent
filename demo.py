"""
Demo Script for Medical AI Agent with RAG Integration
This script demonstrates the agent's capabilities to interface with the RAG system.
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from agent_main import (
    run_medical_assistant, 
    run_medical_research,
    run_interactive_session,
    create_medical_assistant_agent,
    create_medical_query_task
)
from crewai import Crew, Process
from config import get_settings, create_directories
from logger import setup_logging, get_logger

# Setup
setup_logging()
logger = get_logger(__name__)
config = get_settings()


def demo_single_question():
    """Demonstrate answering a single medical question"""
    print("\n" + "=" * 60)
    print("📋 DEMO 1: Single Medical Question")
    print("=" * 60)
    
    question = "What are the common symptoms of diabetes and how is it diagnosed?"
    
    print(f"\n🩺 Question: {question}\n")
    print("-" * 60)
    
    result = run_medical_assistant(question)
    
    print(f"\n🤖 Response:\n{result}")
    print("\n" + "=" * 60)


def demo_medical_research():
    """Demonstrate comprehensive medical research"""
    print("\n" + "=" * 60)
    print("📚 DEMO 2: Medical Research Mode")
    print("=" * 60)
    
    topic = "Hypertension (High Blood Pressure)"
    
    print(f"\n📖 Research Topic: {topic}\n")
    print("-" * 60)
    
    result = run_medical_research(topic)
    
    print(f"\n📄 Research Report:\n{result}")
    print("\n" + "=" * 60)


def demo_multiple_questions():
    """Demonstrate answering multiple questions"""
    print("\n" + "=" * 60)
    print("🔄 DEMO 3: Multiple Medical Questions")
    print("=" * 60)
    
    questions = [
        "What causes chest pain?",
        "How is asthma treated?",
        "What are the symptoms of a urinary tract infection?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n📌 Question {i}: {question}")
        print("-" * 40)
        
        try:
            result = run_medical_assistant(question)
            print(f"\n💡 Answer: {result[:500]}..." if len(result) > 500 else f"\n💡 Answer: {result}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print()
    
    print("=" * 60)


def demo_agent_workflow():
    """Demonstrate the full agent workflow"""
    print("\n" + "=" * 60)
    print("🔧 DEMO 4: Full Agent Workflow")
    print("=" * 60)
    
    print("\n📊 Agent Configuration:")
    print(f"  - Model: {config.AGENT_MODEL}")
    print(f"  - Max Iterations: {config.AGENT_MAX_ITERATIONS}")
    print(f"  - Vector Store: {config.VECTOR_STORE_TYPE}")
    print(f"  - Top K Results: {config.TOP_K_RESULTS}")
    
    print("\n🚀 Creating Medical Assistant Agent...")
    agent = create_medical_assistant_agent()
    
    print(f"  ✓ Agent Role: {agent.role}")
    print(f"  ✓ Tools Available: {[tool.name for tool in agent.tools]}")
    
    question = "A patient presents with burning upon urination that started 1 day ago. What could be the diagnosis and treatment?"
    
    print(f"\n📝 Creating Task for Question:")
    print(f"  '{question[:80]}...'")
    
    task = create_medical_query_task(agent, question)
    
    print("\n⚙️ Executing Crew Workflow...")
    print("-" * 40)
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
        memory=False
    )
    
    result = crew.kickoff()
    
    print("\n" + "-" * 40)
    print(f"\n✅ Final Result:\n{result}")
    print("\n" + "=" * 60)


def quick_test():
    """Quick test to verify everything is working"""
    print("\n" + "=" * 60)
    print("🧪 QUICK TEST")
    print("=" * 60)
    
    print("\n1. Checking environment...")
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        print(f"  ✓ GOOGLE_API_KEY is set (length: {len(api_key)})")
    else:
        print("  ❌ GOOGLE_API_KEY is not set!")
        print("  Please copy .env.example to .env and add your API key")
        return
    
    print("\n2. Testing simple question...")
    try:
        result = run_medical_assistant("What is diabetes?")
        print(f"  ✓ Agent responded successfully")
        print(f"\n  Response preview: {result[:300]}...")
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
    
    print("\n" + "=" * 60)
    print("✅ Quick test completed!")
    print("=" * 60)


def run_full_demo():
    """Run the complete demonstration"""
    print("\n" + "🏥 " * 20)
    print("MEDICAL AI AGENT WITH RAG - DEMONSTRATION")
    print("🏥 " * 20)
    
    print(f"\nSession started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create directories
    create_directories()
    
    # Run demos
    try:
        demo_single_question()
        demo_agent_workflow()
        
    except Exception as e:
        logger.error(f"Demo error: {e}")
        print(f"\n❌ Demo encountered an error: {str(e)}")
    
    print("\n" + "=" * 60)
    print("🎉 DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nTo try the interactive mode, run:")
    print("  python agent_main.py --interactive")
    print("\nOr ask a question directly:")
    print("  python agent_main.py --question 'What are the symptoms of flu?'")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            quick_test()
        elif sys.argv[1] == "--single":
            demo_single_question()
        elif sys.argv[1] == "--research":
            demo_medical_research()
        elif sys.argv[1] == "--workflow":
            demo_agent_workflow()
        elif sys.argv[1] == "--interactive":
            run_interactive_session()
        else:
            run_full_demo()
    else:
        run_full_demo()
