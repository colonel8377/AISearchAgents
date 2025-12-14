#!/usr/bin/env python
"""
Example script demonstrating the Multi-Agent Debate System.

This script shows how to use the debate API without requiring
an actual LLM backend (uses mock responses for demonstration).
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.debate.service import DebateService, generate_default_personas
from src.debate.schemas import PersonaConfig


async def mock_llm_caller(system_prompt: str, user_message: str) -> str:
    """
    Mock LLM caller for demonstration purposes.
    
    In production, replace this with actual LLM API calls.
    """
    # Extract persona style from system prompt
    if "critically evaluate" in system_prompt.lower():
        verdict = 0
        reasoning = "I have serious concerns about this proposal. The evidence is insufficient, and there are significant risks that haven't been adequately addressed."
    elif "strengths" in system_prompt.lower() or "support" in system_prompt.lower():
        verdict = 2
        reasoning = "This is an excellent proposal with strong potential. The benefits clearly outweigh the costs, and I fully support moving forward."
    else:
        verdict = 1
        reasoning = "The proposal has both merits and drawbacks. We should proceed cautiously while addressing the identified concerns."
    
    return f'{{"verdict": {verdict}, "reasoning": "{reasoning}"}}'


async def run_debate_example():
    """Run a complete debate example."""
    
    print("=" * 70)
    print("Multi-Agent Debate System - Example")
    print("=" * 70)
    
    # Initialize the service
    service = DebateService()
    service.set_llm_caller(mock_llm_caller)
    
    # Define the debate topic
    topic = "Should we invest heavily in renewable energy infrastructure?"
    print(f"\n📋 Topic: {topic}\n")
    
    # Create custom personas
    personas = [
        PersonaConfig(
            name="Dr. Green",
            description="Environmental scientist passionate about sustainability",
            style="Supportive"
        ),
        PersonaConfig(
            name="Ms. Pragmatic",
            description="Policy analyst focused on practical implementation",
            style="Neutral"
        ),
        PersonaConfig(
            name="Mr. Skeptic",
            description="Financial advisor concerned about economic viability",
            style="Critical"
        )
    ]
    
    # Create the debate session
    session_id, agents = service.create_session(topic, personas)
    print(f"✅ Created debate session: {session_id}\n")
    print(f"👥 Participants:")
    for agent in agents:
        print(f"   - {agent.role_name} (ID: {str(agent.agent_id)[:8]}...)")
    
    # Run multiple rounds
    max_rounds = 5
    print(f"\n🔄 Starting debate (max {max_rounds} rounds)...\n")
    
    for round_num in range(max_rounds):
        print(f"\n{'='*70}")
        print(f"Round {round_num + 1}")
        print('='*70)
        
        # Prepare context
        if round_num == 0:
            context = "This is the opening round. Please share your initial position on the topic."
        else:
            context = f"Round {round_num + 1}. Consider the diverse perspectives shared in previous rounds."
        
        # Collect votes from all agents
        votes = []
        for agent in agents:
            # Call the agent
            response = await service.call_llm(agent.system_prompt, f"Topic: {topic}\n\nContext: {context}")
            
            # Parse response (simplified for demo)
            import json
            try:
                parsed = json.loads(response)
                verdict = parsed['verdict']
                reasoning = parsed['reasoning']
            except:
                verdict = 1
                reasoning = response
            
            votes.append(verdict)
            
            # Display agent response
            print(f"\n{agent.role_name}:")
            print(f"  Verdict: {verdict}")
            print(f"  Reasoning: {reasoning[:200]}...")
        
        # Add votes to history
        service.add_vote_round(session_id, votes)
        
        # Check stability
        is_stable = service.calculate_stability(session_id)
        
        print(f"\n📊 Vote Distribution: {votes}")
        print(f"🎯 Stability: {'✅ STABLE' if is_stable else '❌ NOT STABLE'}")
        
        if is_stable:
            print(f"\n{'='*70}")
            print(f"✨ Debate reached consensus after {round_num + 1} rounds!")
            print(f"{'='*70}")
            break
    else:
        print(f"\n{'='*70}")
        print(f"⏱️  Debate did not converge within {max_rounds} rounds")
        print(f"{'='*70}")
    
    # Summary
    session = service.get_session(session_id)
    print(f"\n📈 Final Statistics:")
    print(f"   Total rounds: {len(session['vote_history'])}")
    print(f"   Vote history: {session['vote_history']}")
    print(f"   Agents: {len(agents)}")


async def main():
    """Main entry point."""
    try:
        await run_debate_example()
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
