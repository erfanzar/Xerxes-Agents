#!/usr/bin/env python3
"""
Scenario 1: Conversational AI Assistant with Memory
A helpful assistant that remembers context and learns from interactions.
"""

import asyncio
import os
from pathlib import Path

import openai

from calute import Agent, AssistantMessage, Calute, MessagesHistory, UserMessage
from calute.config import CaluteConfig, set_config
from calute.memory import MemoryStore, MemoryType

# Configure the system
config = CaluteConfig(
    environment="production",
    executor={"default_timeout": 30.0, "max_retries": 2},
    memory={"max_short_term": 50, "max_working": 10},
    logging={"level": "INFO"},
)
set_config(config)

# Initialize OpenAI client with xe-1 model
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "YOUR-KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", None),
)

# Initialize enhanced memory
memory_store = MemoryStore(
    max_short_term=100,
    max_working=20,
    enable_persistence=True,
    persistence_path=Path.home() / ".calute" / "conversation_memory",
)


def search_knowledge(query: str) -> str:
    """Search internal knowledge base."""
    # Simulate knowledge search
    knowledge = {
        "python": "Python is a high-level programming language known for readability.",
        "calute": "Calute is an AI agent orchestration framework.",
        "memory": "Memory systems help AI retain and recall information.",
    }

    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return f"Knowledge: {value}"

    return "No specific knowledge found. Please provide more details."


def save_user_preference(preference: str, value: str) -> str:
    """Save user preferences for personalization."""
    # Add to memory
    memory_store.add_memory(
        content=f"User preference: {preference} = {value}",
        memory_type=MemoryType.LONG_TERM,
        agent_id="assistant",
        tags=["preference", "user_profile"],
        importance_score=0.9,
    )
    return f"Saved preference: {preference} = {value}"


def recall_conversation_context(topic: str = "") -> str:
    """Recall previous conversation context."""
    memories = memory_store.retrieve_memories(tags=["conversation", topic] if topic else ["conversation"], limit=5)

    if not memories:
        return "No previous context found."

    context = "Previous context:\n"
    for mem in memories:
        context += f"- {mem.content}\n"

    return context


def analyze_sentiment(text: str) -> str:
    """Analyze sentiment of user input."""
    # Simple sentiment analysis
    positive_words = ["happy", "great", "excellent", "good", "love", "wonderful"]
    negative_words = ["sad", "bad", "terrible", "hate", "awful", "horrible"]

    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)

    if positive_count > negative_count:
        sentiment = "positive"
    elif negative_count > positive_count:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    # Store sentiment in memory
    memory_store.add_memory(
        content=f"User sentiment: {sentiment} - {text[:50]}",
        memory_type=MemoryType.EPISODIC,
        agent_id="assistant",
        tags=["sentiment", sentiment],
        importance_score=0.5,
    )

    return f"Detected sentiment: {sentiment}"


async def main():
    """Run the conversational assistant scenario."""
    print("=" * 60)
    print("🤖 CONVERSATIONAL AI ASSISTANT WITH MEMORY")
    print("=" * 60)
    print()

    # Create the conversational agent
    agent = Agent(
        id="conversational_assistant",
        name="Memory-Enhanced Assistant",
        model="gpt-4o",
        instructions="""You are a helpful, friendly assistant with perfect memory.
        You remember user preferences, past conversations, and learn from interactions.
        Use your functions to:
        1. Search knowledge when asked questions
        2. Save user preferences when mentioned
        3. Recall context from previous conversations
        4. Analyze sentiment to understand user mood

        Be personable and reference past interactions when relevant.""",
        functions=[search_knowledge, save_user_preference, recall_conversation_context, analyze_sentiment],
        max_tokens=500,
        temperature=0.7,
    )

    # Initialize Calute
    calute = Calute(client, enable_memory=True)
    calute.memory = memory_store
    calute.register_agent(agent)

    # Simulate a conversation
    messages = MessagesHistory(messages=[])

    conversations = [
        "Hello! I'm John and I love Python programming.",
        "Can you tell me about Python?",
        "I prefer dark mode for coding, please remember that.",
        "What do you know about Calute?",
        "Do you remember what I told you about my preferences?",
        "I'm feeling really happy today because I solved a difficult bug!",
        "Can you recall our conversation so far?",
    ]

    print("Starting conversation...\n")

    for user_input in conversations:
        print(f"👤 User: {user_input}")

        # Add to messages
        messages.messages.append(UserMessage(content=user_input))

        # Store conversation in memory
        memory_store.add_memory(
            content=f"User said: {user_input}",
            memory_type=MemoryType.SHORT_TERM,
            agent_id="assistant",
            tags=["conversation", "user_input"],
            importance_score=0.6,
        )

        # Get response
        try:
            response = await calute.create_response(
                prompt=user_input,
                messages=messages,
                agent_id=agent.id,
                stream=False,
                apply_functions=True,
                use_instructed_prompt=True,
            )

            # Extract response text
            if hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)

            print(f"🤖 Assistant: {response_text}")

            # Store response in memory
            memory_store.add_memory(
                content=f"Assistant responded: {response_text[:100]}",
                memory_type=MemoryType.SHORT_TERM,
                agent_id="assistant",
                tags=["conversation", "assistant_response"],
                importance_score=0.5,
            )

            # Add to message history
            messages.messages.append(AssistantMessage(content=response_text))

        except Exception as e:
            print(f"Error: {e}")
            # Fallback response
            response_text = "I understand. Let me help you with that."
            print(f"🤖 Assistant: {response_text}")
            messages.messages.append(AssistantMessage(content=response_text))

        print()
        await asyncio.sleep(1)  # Small delay between messages

    # Show memory statistics
    print("\n" + "=" * 60)
    print("📊 MEMORY STATISTICS")
    print("=" * 60)

    stats = memory_store.get_statistics()
    print(f"Total memories: {stats['total_memories']}")
    print(f"Memory types: {stats['by_type']}")
    print(f"Cache performance: {stats['cache_hit_rate']:.1%} hit rate")

    # Show learned preferences
    print("\n📝 Learned User Preferences:")
    preferences = memory_store.retrieve_memories(tags=["preference"], limit=10)
    for pref in preferences:
        print(f"  - {pref.content}")

    # Show sentiment analysis
    print("\n😊 Sentiment Analysis:")
    sentiments = memory_store.retrieve_memories(tags=["sentiment"], limit=5)
    for sent in sentiments:
        print(f"  - {sent.content}")

    # Save memory for next session
    memory_store.save()
    print("\n💾 Memory saved for next session!")

    print("\n✅ Conversational assistant scenario completed!")


if __name__ == "__main__":
    asyncio.run(main())
