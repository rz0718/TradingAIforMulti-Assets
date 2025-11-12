#!/usr/bin/env python3
"""
Test script for OpenRouter API connection.
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def test_openrouter_connection():
    """Test OpenRouter API connection with Claude and Gemini."""

    # Get API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment variables")
        return False

    print(f"✅ Found API key (length: {len(api_key)})")

    # Initialize OpenRouter client
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", max_retries=0)

    # Test models
    models_to_test = ["anthropic/claude-sonnet-4.5", "google/gemini-2.5-pro"]

    for model in models_to_test:
        print(f"\n🧪 Testing {model}...")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": "Hello! Please respond with just 'OK' to test the connection.",
                    }
                ],
                max_tokens=10,
                temperature=0.1,
            )

            content = response.choices[0].message.content
            print(f"✅ {model}: Response received - '{content}'")

        except Exception as e:
            print(f"❌ {model}: Error - {e}")
            return False

    print("\n🎉 All OpenRouter connections successful!")
    return True


if __name__ == "__main__":
    test_openrouter_connection()
