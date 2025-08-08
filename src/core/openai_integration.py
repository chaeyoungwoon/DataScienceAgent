"""
OpenAI Integration Module

Provides wrapper functions for OpenAI LLM calls with structured output.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
import openai
from datetime import datetime

logger = logging.getLogger(__name__)


class OpenAIIntegration:
    """
    OpenAI integration for structured LLM calls.
    
    Provides helper functions for code generation, task analysis,
    search query generation, dataset relevance scoring, insight
    generation, and documentation generation.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI integration.
        
        Args:
            api_key: OpenAI API key (if None, will try to get from environment)
            model_name: OpenAI model to use
        """
        self.model_name = model_name
        self.api_key = api_key
        
        # Setup OpenAI client
        if self.api_key:
            openai.api_key = self.api_key
        else:
            # Try to get from environment
            import os
            self.api_key = os.getenv('OPENAI_API_KEY')
            if self.api_key:
                openai.api_key = self.api_key
        
        if not self.api_key:
            logger.warning("OpenAI API key not provided")
            self.client = None
        else:
            try:
                self.client = openai.OpenAI(api_key=self.api_key)
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
    
    async def generate_code(self, task_description: str, context: Dict[str, Any] = None, language: str = "python") -> Dict[str, Any]:
        """
        Generate code based on task description.
        
        Args:
            task_description: Description of the code to generate
            context: Additional context for code generation
            language: Programming language for code generation
            
        Returns:
            Dictionary containing generated code and metadata
        """
        if not self.client:
            return {
                'code': f"# {task_description}\n# OpenAI not available",
                'language': language,
                'status': 'error',
                'error': 'OpenAI client not available'
            }
        
        try:
            prompt = f"""
            Generate {language} code for the following task:
            
            Task: {task_description}
            
            Context: {json.dumps(context, indent=2) if context else 'None'}
            
            Requirements:
            1. Generate clean, well-documented code
            2. Include proper error handling
            3. Follow best practices for {language}
            4. Include comments explaining the logic
            
            Return only the code, no explanations.
            """
            
            response = await self._async_generate_text(prompt)
            
            return {
                'code': response,
                'language': language,
                'status': 'success',
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return {
                'code': f"# Error: {str(e)}",
                'language': language,
                'status': 'error',
                'error': str(e)
            }
    
    async def analyze_data_science_task(self, research_question: str, data_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze a data science task and provide recommendations.
        
        Args:
            research_question: The research question or task
            data_context: Context about the data and requirements
            
        Returns:
            Dictionary containing analysis and recommendations
        """
        if not self.client:
            return {
                'analysis': f"Task: {research_question}",
                'recommendations': ['OpenAI not available'],
                'status': 'error',
                'error': 'OpenAI client not available'
            }
        
        try:
            prompt = f"""
            Analyze this data science task and provide recommendations:
            
            Research Question: {research_question}
            
            Data Context: {json.dumps(data_context, indent=2) if data_context else 'None'}
            
            Please provide:
            1. Task analysis and understanding
            2. Recommended approach
            3. Key considerations
            4. Potential challenges
            5. Success metrics
            
            Format as JSON with keys: analysis, approach, considerations, challenges, metrics
            """
            
            response = await self._async_generate_text(prompt)
            
            # Try to parse JSON response
            try:
                parsed_response = json.loads(response)
                return {
                    'analysis': parsed_response.get('analysis', ''),
                    'approach': parsed_response.get('approach', ''),
                    'considerations': parsed_response.get('considerations', []),
                    'challenges': parsed_response.get('challenges', []),
                    'metrics': parsed_response.get('metrics', []),
                    'status': 'success',
                    'timestamp': datetime.utcnow().isoformat()
                }
            except json.JSONDecodeError:
                return {
                    'analysis': response,
                    'approach': 'Manual analysis required',
                    'considerations': ['Parse response manually'],
                    'challenges': ['Response format issue'],
                    'metrics': ['TBD'],
                    'status': 'partial',
                    'timestamp': datetime.utcnow().isoformat()
                }
            
        except Exception as e:
            logger.error(f"Task analysis failed: {e}")
            return {
                'analysis': f"Error: {str(e)}",
                'approach': 'Error occurred',
                'considerations': [],
                'challenges': [str(e)],
                'metrics': [],
                'status': 'error',
                'error': str(e)
            }
    
    async def generate_search_queries(self, research_question: str, domain: str = "general", max_queries: int = 10) -> List[str]:
        """
        Generate search queries for dataset discovery.
        
        Args:
            research_question: The research question
            domain: Domain context (e.g., "healthcare", "finance")
            max_queries: Maximum number of queries to generate
            
        Returns:
            List of search queries
        """
        if not self.client:
            return [f"{domain} dataset {research_question}"]
        
        try:
            prompt = f"""
            Generate {max_queries} search queries for finding datasets related to this research question:
            
            Research Question: {research_question}
            Domain: {domain}
            
            Generate diverse search queries that could help find relevant datasets.
            Return only the queries, one per line, no numbering or formatting.
            """
            
            response = await self._async_generate_text(prompt)
            
            # Parse queries from response
            queries = [line.strip() for line in response.split('\n') if line.strip()]
            
            # Limit to max_queries
            queries = queries[:max_queries]
            
            # Add fallback if no queries generated
            if not queries:
                queries = [f"{domain} dataset {research_question}"]
            
            return queries
            
        except Exception as e:
            logger.error(f"Search query generation failed: {e}")
            return [f"{domain} dataset {research_question}"]
    
    async def score_dataset_relevance(self, dataset_description: str, research_question: str, requirements: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Score dataset relevance to research question.
        
        Args:
            dataset_description: Description of the dataset
            research_question: The research question
            requirements: Additional requirements
            
        Returns:
            Dictionary containing relevance score and reasoning
        """
        if not self.client:
            return {
                'score': 0.5,
                'reasoning': 'OpenAI not available',
                'confidence': 0.0,
                'status': 'error'
            }
        
        try:
            prompt = f"""
            Score the relevance of this dataset to the research question:
            
            Dataset: {dataset_description}
            Research Question: {research_question}
            Requirements: {json.dumps(requirements, indent=2) if requirements else 'None'}
            
            Provide a relevance score from 0.0 to 1.0 and reasoning.
            Format as JSON with keys: score, reasoning, confidence
            """
            
            response = await self._async_generate_text(prompt)
            
            # Try to parse JSON response
            try:
                parsed_response = json.loads(response)
                return {
                    'score': float(parsed_response.get('score', 0.5)),
                    'reasoning': parsed_response.get('reasoning', ''),
                    'confidence': float(parsed_response.get('confidence', 0.5)),
                    'status': 'success',
                    'timestamp': datetime.utcnow().isoformat()
                }
            except (json.JSONDecodeError, ValueError):
                return {
                    'score': 0.5,
                    'reasoning': response,
                    'confidence': 0.5,
                    'status': 'partial',
                    'timestamp': datetime.utcnow().isoformat()
                }
            
        except Exception as e:
            logger.error(f"Dataset scoring failed: {e}")
            return {
                'score': 0.5,
                'reasoning': f"Error: {str(e)}",
                'confidence': 0.0,
                'status': 'error',
                'error': str(e)
            }
    
    async def generate_insights(self, analysis_results: Dict[str, Any], research_question: str) -> Dict[str, Any]:
        """
        Generate insights from analysis results.
        
        Args:
            analysis_results: Results from data analysis
            research_question: The original research question
            
        Returns:
            Dictionary containing generated insights
        """
        if not self.client:
            return {
                'insights': ['OpenAI not available'],
                'summary': 'Manual analysis required',
                'status': 'error'
            }
        
        try:
            prompt = f"""
            Generate insights from this analysis:
            
            Research Question: {research_question}
            Analysis Results: {json.dumps(analysis_results, indent=2)}
            
            Provide:
            1. Key insights (list)
            2. Summary of findings
            3. Implications
            4. Recommendations
            
            Format as JSON with keys: insights, summary, implications, recommendations
            """
            
            response = await self._async_generate_text(prompt)
            
            # Try to parse JSON response
            try:
                parsed_response = json.loads(response)
                return {
                    'insights': parsed_response.get('insights', []),
                    'summary': parsed_response.get('summary', ''),
                    'implications': parsed_response.get('implications', []),
                    'recommendations': parsed_response.get('recommendations', []),
                    'status': 'success',
                    'timestamp': datetime.utcnow().isoformat()
                }
            except json.JSONDecodeError:
                return {
                    'insights': [response],
                    'summary': 'Manual parsing required',
                    'implications': [],
                    'recommendations': [],
                    'status': 'partial',
                    'timestamp': datetime.utcnow().isoformat()
                }
            
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            return {
                'insights': [f"Error: {str(e)}"],
                'summary': 'Error occurred',
                'implications': [],
                'recommendations': [],
                'status': 'error',
                'error': str(e)
            }
    
    async def generate_documentation(self, project_results: Dict[str, Any], format_type: str = "markdown") -> str:
        """
        Generate project documentation.
        
        Args:
            project_results: Results from the project
            format_type: Output format (markdown, html, etc.)
            
        Returns:
            Generated documentation as string
        """
        if not self.client:
            return f"# Project Documentation\n\nOpenAI not available\n\nResults: {json.dumps(project_results, indent=2)}"
        
        try:
            prompt = f"""
            Generate {format_type} documentation for this data science project:
            
            Project Results: {json.dumps(project_results, indent=2)}
            
            Create comprehensive documentation including:
            1. Executive summary
            2. Methodology
            3. Results and findings
            4. Conclusions
            5. Recommendations
            
            Format as {format_type}.
            """
            
            response = await self._async_generate_text(prompt)
            return response
            
        except Exception as e:
            logger.error(f"Documentation generation failed: {e}")
            return f"# Project Documentation\n\nError: {str(e)}\n\nResults: {json.dumps(project_results, indent=2)}"
    
    async def _async_generate_text(self, prompt: str) -> str:
        """
        Asynchronously call OpenAI API.
        
        Args:
            prompt: The prompt to send to OpenAI
            
        Returns:
            Generated text response
        """
        if not self.client:
            return "OpenAI client not available"
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant for data science tasks."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return f"Error: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if OpenAI integration is available."""
        return self.client is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the OpenAI model."""
        return {
            'model_name': self.model_name,
            'available': self.is_available(),
            'api_key_configured': self.api_key is not None
        }
