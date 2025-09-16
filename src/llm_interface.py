# LLM interface module for OpenAI API integration.
# Handles query processing, context assembly, and response generation.

import logging
import time
from typing import List, Dict, Any, Optional
import json

import openai
from openai import OpenAI
import os

from utils import TokenCounter, PerformanceTracker, validate_input
from exceptions import LLMError, APIError, TokenLimitError


class LLMInterface:
    #OpenAI API interface for generating responses with retrieved context

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config.get('models', {})

        # Model configuration
        self.model_name = self.model_config.get('llm_model', 'gpt-3.5-turbo')
        self.temperature = self.model_config.get('llm_temperature', 0.1)
        self.max_tokens = self.model_config.get('llm_max_tokens', 1000)

        # API configuration
        # self.api_key = config.get('api_keys', {}).get('openai')
        self.api_key = 'YOUR_KEY_HERE'

        # Evaluation configuration
        self.track_usage = config.get('evaluation', {}).get('track_token_usage', True)

        # Initialize components
        self.client = None
        self.token_counter = TokenCounter(self.model_name)
        self.performance_tracker = PerformanceTracker()
        self.usage_stats = {
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'total_requests': 0,
            'total_cost_estimate': 0.0
        }

        self.logger = logging.getLogger(__name__)

        # Initialize OpenAI client
        self._initialize_client()

    def _initialize_client(self):
        #Initialize OpenAI client with API key
        try:
            if not self.api_key:
                raise APIError("OpenAI API key not provided")

            self.client = OpenAI(api_key=self.api_key)

            # Test the connection with a minimal request
            self.logger.info("Testing OpenAI API connection...")

            test_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
                temperature=0
            )

            self.logger.info(f"Successfully connected to OpenAI API using model: {self.model_name}")

        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            raise APIError(f"OpenAI initialization failed: {e}") from e

    #   Generate response using query and retrieved documents.
    def generate_response(
            self,
            query: str, #   User's question
            retrieved_docs: List[Dict[str, Any]], # List of retrieved documents with metadata
            include_sources: bool = True, # include source information in response
    ) -> Dict[str, Any]:
        #   Returns dictionary containing response and metadata

        try:
            validate_input(query, str, "query")
            validate_input(retrieved_docs, list, "retrieved_docs")

            if not query.strip():
                raise LLMError("Cannot generate response for empty query")

            self.logger.info(f"Generating response for query: {query[:100]}...")
            self.performance_tracker.start_timer('response_generation')

            # Assemble context from retrieved documents
            context = self._assemble_context(retrieved_docs)

            # Create the prompt
            prompt = self._create_prompt(query, context)

            # Check token limits
            self._validate_token_limits(prompt)

            # Generate response using OpenAI
            response_data = self._call_openai_api(prompt)

            # Process and format response
            formatted_response = self._format_response(
                query,
                response_data,
                retrieved_docs,
                include_sources
            )

            generation_time = self.performance_tracker.end_timer('response_generation')
            formatted_response['generation_time'] = generation_time

            self.logger.info(f"Response generated successfully in {generation_time:.2f}s")

            return formatted_response

        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            raise LLMError(f"Failed to generate response: {e}") from e

    def _assemble_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        #Assemble context from retrieved documents
        if not retrieved_docs:
            return "No relevant context found."

        context_parts = []

        for i, doc in enumerate(retrieved_docs, 1):
            # Extract document information
            text = doc.get('document', '')
            metadata = doc.get('metadata', {})
            similarity = doc.get('similarity', 0.0)

            # Get source information
            source_file = metadata.get('source_file', 'Unknown')
            page_numbers = metadata.get('page_numbers', [])

            # Format page information
            page_info = ""
            if page_numbers:
                if len(page_numbers) == 1:
                    page_info = f" (Page {page_numbers[0]})"
                else:
                    page_info = f" (Pages {page_numbers[0]}-{page_numbers[-1]})"

            # Create context entry
            context_entry = (
                f"Source {i} [{source_file}{page_info}] (Relevance: {similarity:.2f}):\n"
                f"{text}\n"
            )

            context_parts.append(context_entry)

        return "\n".join(context_parts)

    def _create_prompt(self, query: str, context: str) -> List[Dict[str, str]]:
        #Create structured prompt for the LLM

        system_prompt = """You are a knowledgeable assistant helping users understand academic documents. 
        Your task is to answer questions based on the provided context from textbooks and research papers.

        Instructions:
        1. Answer the question using ONLY the information provided in the context
        2. If the context doesn't contain enough information to answer the question, say so clearly
        3. Be precise and cite specific sources when making claims
        4. If multiple sources contain relevant information, synthesize them appropriately
        5. Maintain academic rigor and accuracy in your responses
        6. If you notice contradictions between sources, point them out
        7. DO NOT HALLUCINATE

        Format your response clearly and include relevant details from the context."""

        user_prompt = f"""Based on the following context from academic documents, please answer this question:

        Question: {query}

        Context:
        {context}

        Please provide a comprehensive answer based on the context provided."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def _validate_token_limits(self, messages: List[Dict[str, str]]):
        #Validate that the prompt doesn't exceed model token limits
        # Calculate token count for the entire conversation
        total_tokens = sum(
            self.token_counter.count_tokens(msg['content'])
            for msg in messages
        )

        # Add buffer for response tokens
        total_with_response = total_tokens + self.max_tokens

        # Model-specific limits (approximate)
        model_limits = {
            'gpt-3.5-turbo': 4096,
            'gpt-3.5-turbo-16k': 16384,
            'gpt-4': 8192,
            'gpt-4-32k': 32768,
            'gpt-4-turbo': 128000,
            'gpt-4o': 128000
        }

        limit = model_limits.get(self.model_name, 4096)

        if total_with_response > limit:
            self.logger.warning(
                f"Token count ({total_with_response}) exceeds model limit ({limit})"
            )
            # Note: In a production system, you might want to truncate context here
            # For learning purposes, we'll log a warning but continue

        self.logger.debug(f"Prompt tokens: {total_tokens}, Model limit: {limit}")

    def _call_openai_api(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        #Make API call to OpenAI
        try:
            self.logger.debug("Calling OpenAI API...")

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )

            # Track usage statistics
            if self.track_usage and response.usage:
                self._update_usage_stats(response.usage)

            return {
                'content': response.choices[0].message.content,
                'model': response.model,
                'usage': response.usage.dict() if response.usage else None,
                'finish_reason': response.choices[0].finish_reason
            }

        except openai.RateLimitError as e:
            self.logger.error(f"OpenAI rate limit exceeded: {e}")
            raise APIError(f"Rate limit exceeded: {e}") from e

        except openai.AuthenticationError as e:
            self.logger.error(f"OpenAI authentication failed: {e}")
            raise APIError(f"Authentication failed: {e}") from e

        except openai.APIError as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise APIError(f"API error: {e}") from e

        except Exception as e:
            self.logger.error(f"Unexpected error calling OpenAI API: {e}")
            raise APIError(f"API call failed: {e}") from e

    def _format_response(
            self,
            query: str,
            response_data: Dict[str, Any],
            retrieved_docs: List[Dict[str, Any]],
            include_sources: bool
    ) -> Dict[str, Any]:
        #Format the final response with metadata

        formatted_response = {
            'query': query,
            'answer': response_data['content'],
            'model_used': response_data['model'],
            'finish_reason': response_data['finish_reason'],
            'sources_count': len(retrieved_docs),
            'usage': response_data.get('usage', {}),
            'timestamp': time.time()
        }

        if include_sources:
            # Format source information
            sources = []
            for i, doc in enumerate(retrieved_docs, 1):
                metadata = doc.get('metadata', {})
                source_info = {
                    'source_number': i,
                    'file': metadata.get('source_file', 'Unknown'),
                    'pages': metadata.get('page_numbers', []),
                    'similarity': doc.get('similarity', 0.0),
                    'chunk_id': metadata.get('chunk_id', ''),
                    'preview': doc.get('document', '')[:200] + '...' if len(doc.get('document', '')) > 200 else doc.get(
                        'document', '')
                }
                sources.append(source_info)

            formatted_response['sources'] = sources

        return formatted_response

    def _update_usage_stats(self, usage):
        #Update token usage statistics
        self.usage_stats['total_prompt_tokens'] += usage.prompt_tokens
        self.usage_stats['total_completion_tokens'] += usage.completion_tokens
        self.usage_stats['total_requests'] += 1

        # Rough cost estimation (prices as of 2024)
        cost_per_1k_prompt = 0.0015  # GPT-3.5-turbo input cost
        cost_per_1k_completion = 0.002  # GPT-3.5-turbo output cost

        prompt_cost = (usage.prompt_tokens / 1000) * cost_per_1k_prompt
        completion_cost = (usage.completion_tokens / 1000) * cost_per_1k_completion

        self.usage_stats['total_cost_estimate'] += prompt_cost + completion_cost

    def get_usage_stats(self) -> Dict[str, Any]:
        #Get current usage statistics
        return {
            **self.usage_stats,
            'average_prompt_tokens': (
                    self.usage_stats['total_prompt_tokens'] / max(1, self.usage_stats['total_requests'])
            ),
            'average_completion_tokens': (
                    self.usage_stats['total_completion_tokens'] / max(1, self.usage_stats['total_requests'])
            )
        }

    def reset_usage_stats(self):
        #Reset usage statistics
        self.usage_stats = {
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'total_requests': 0,
            'total_cost_estimate': 0.0
        }
        self.logger.info("Usage statistics reset")

    def get_performance_stats(self) -> Dict[str, Any]:
        #Get performance statistics
        return {
            'performance_metrics': self.performance_tracker.metrics,
            'usage_stats': self.get_usage_stats(),
            'model_config': {
                'model_name': self.model_name,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens
            }
        }