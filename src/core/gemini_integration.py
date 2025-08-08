"""
Google Gemini LLM Integration for Data Science Agent Swarm

This module provides integration with Google's Gemini LLM for enhanced
agent capabilities including code generation, analysis, and reasoning.
"""

import os
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
import google.generativeai as genai
from google.generativeai import GenerativeModel
import json
import re

class GeminiIntegration:
    """
    Integration class for Google Gemini LLM.
    
    Provides methods for:
    - Code generation for data science tasks
    - Analysis and reasoning
    - Documentation generation
    - Task planning and decomposition
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        """
        Initialize Gemini integration.
        
        Args:
            api_key: Gemini API key for Gemini access
            model_name: Gemini model to use (gemini-pro, gemini-pro-vision)
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        
        # Set up API key
        self.api_key = api_key
        if api_key:
            genai.configure(api_key=api_key)
        else:
            # Try to get from environment
            self.api_key = os.getenv('GEMINI_API_KEY')
            if self.api_key:
                genai.configure(api_key=self.api_key)
            else:
                self.logger.warning("No Gemini API key found. Some features may not work.")
        
        # Initialize the model
        try:
            self.model = GenerativeModel(model_name)
            self.logger.info(f"Gemini model '{model_name}' initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini model: {e}")
            self.model = None
    
    async def generate_code(
        self, 
        task_description: str, 
        context: Dict[str, Any] = None,
        language: str = "python"
    ) -> Dict[str, Any]:
        """
        Generate code for a specific data science task.
        
        Args:
            task_description: Description of the task to generate code for
            context: Additional context about data, requirements, etc.
            language: Programming language for code generation
            
        Returns:
            Dictionary containing generated code and metadata
        """
        if not self.model:
            return {"error": "Gemini model not initialized"}
        
        try:
            # Build prompt for code generation
            prompt = self._build_code_generation_prompt(task_description, context, language)
            
            # Generate code
            response = await self._async_generate_text(prompt)
            
            # Extract code from response
            code_blocks = self._extract_code_blocks(response)
            
            return {
                "code": code_blocks,
                "explanation": self._extract_explanation(response),
                "language": language,
                "task": task_description,
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Error generating code: {e}")
            return {"error": str(e), "status": "error"}
    
    async def analyze_data_science_task(
        self, 
        research_question: str,
        data_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze a data science research question and provide recommendations.
        
        Args:
            research_question: The research question to analyze
            data_context: Information about available data
            
        Returns:
            Analysis results with recommendations
        """
        if not self.model:
            return {"error": "Gemini model not initialized"}
        
        try:
            prompt = self._build_analysis_prompt(research_question, data_context)
            response = await self._async_generate_text(prompt)
            
            # Parse structured response
            analysis = self._parse_analysis_response(response)
            
            return {
                "analysis": analysis,
                "research_question": research_question,
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing task: {e}")
            return {"error": str(e), "status": "error"}
    
    async def generate_search_queries(
        self, 
        research_question: str, 
        domain: str = "general",
        max_queries: int = 10
    ) -> List[str]:
        """
        Generate search queries for dataset discovery.
        
        Args:
            research_question: The research question
            domain: Domain of the research
            max_queries: Maximum number of queries to generate
            
        Returns:
            List of search queries
        """
        if not self.model:
            return []
        
        try:
            prompt = f"""
            Generate {max_queries} diverse search queries for finding datasets related to this research question: "{research_question}"
            Domain: {domain}
            
            The queries should:
            1. Cover different aspects of the research question
            2. Use various relevant keywords
            3. Include domain-specific terms
            4. Be suitable for searching on Kaggle, GitHub, and other data sources
            
            Return only the queries, one per line, without numbering.
            """
            
            response = await self._async_generate_text(prompt)
            
            # Extract queries from response
            queries = [line.strip() for line in response.split('\n') if line.strip()]
            
            # Limit to requested number
            return queries[:max_queries]
            
        except Exception as e:
            self.logger.error(f"Error generating search queries: {e}")
            return []
    
    async def score_dataset_relevance(
        self, 
        dataset_description: str, 
        research_question: str,
        requirements: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Score dataset relevance using Gemini's understanding.
        
        Args:
            dataset_description: Description of the dataset
            research_question: The research question
            requirements: Data requirements
            
        Returns:
            Relevance scoring results
        """
        if not self.model:
            return {"relevance_score": 0.0, "reasoning": "Model not available"}
        
        try:
            prompt = f"""
            Analyze the relevance of this dataset for the given research question.
            
            Research Question: {research_question}
            Dataset Description: {dataset_description}
            
            Requirements: {requirements or 'None specified'}
            
            Please provide:
            1. A relevance score from 0.0 to 1.0
            2. Detailed reasoning for the score
            3. Key factors that make this dataset relevant or irrelevant
            4. Suggestions for how to use this dataset
            
            Format your response as JSON with keys: score, reasoning, factors, suggestions
            """
            
            response = await self._async_generate_text(prompt)
            
            # Try to parse JSON response
            try:
                result = json.loads(response)
                return {
                    "relevance_score": float(result.get("score", 0.0)),
                    "reasoning": result.get("reasoning", ""),
                    "factors": result.get("factors", []),
                    "suggestions": result.get("suggestions", []),
                    "status": "success"
                }
            except json.JSONDecodeError:
                # Fallback to text parsing
                return self._parse_relevance_response(response)
                
        except Exception as e:
            self.logger.error(f"Error scoring dataset relevance: {e}")
            return {"relevance_score": 0.0, "reasoning": f"Error: {str(e)}"}
    
    async def generate_insights(
        self, 
        analysis_results: Dict[str, Any],
        research_question: str
    ) -> Dict[str, Any]:
        """
        Generate insights from analysis results.
        
        Args:
            analysis_results: Results from data analysis
            research_question: Original research question
            
        Returns:
            Generated insights and recommendations
        """
        if not self.model:
            return {"error": "Gemini model not initialized"}
        
        try:
            prompt = f"""
            Based on the following analysis results, generate insights and recommendations for this research question: "{research_question}"
            
            Analysis Results:
            {json.dumps(analysis_results, indent=2)}
            
            Please provide:
            1. Key insights (3-5 main findings)
            2. Statistical significance of findings
            3. Practical implications
            4. Recommendations for next steps
            5. Limitations and caveats
            
            Format as JSON with keys: insights, significance, implications, recommendations, limitations
            """
            
            response = await self._async_generate_text(prompt)
            
            try:
                result = json.loads(response)
                return {
                    "insights": result.get("insights", []),
                    "significance": result.get("significance", ""),
                    "implications": result.get("implications", []),
                    "recommendations": result.get("recommendations", []),
                    "limitations": result.get("limitations", []),
                    "status": "success"
                }
            except json.JSONDecodeError:
                return {
                    "insights": [response],
                    "status": "success"
                }
                
        except Exception as e:
            self.logger.error(f"Error generating insights: {e}")
            return {"error": str(e), "status": "error"}
    
    async def generate_documentation(
        self, 
        project_results: Dict[str, Any],
        format_type: str = "markdown"
    ) -> str:
        """
        Generate comprehensive documentation for research project.
        
        Args:
            project_results: Complete project results
            format_type: Output format (markdown, html, latex)
            
        Returns:
            Generated documentation
        """
        if not self.model:
            return "Error: Gemini model not initialized"
        
        try:
            prompt = f"""
            Generate comprehensive {format_type} documentation for this data science research project.
            
            Project Results:
            {json.dumps(project_results, indent=2)}
            
            The documentation should include:
            1. Executive Summary
            2. Methodology
            3. Data Analysis Results
            4. Model Performance
            5. Key Findings
            6. Recommendations
            7. Technical Appendix
            
            Make it publication-ready and professional.
            """
            
            response = await self._async_generate_text(prompt)
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating documentation: {e}")
            return f"Error generating documentation: {str(e)}"
    
    def _build_code_generation_prompt(
        self, 
        task_description: str, 
        context: Dict[str, Any] = None,
        language: str = "python"
    ) -> str:
        """Build prompt for code generation."""
        prompt = f"""
        Generate {language} code for the following data science task:
        
        Task: {task_description}
        
        Context: {context or 'No additional context provided'}
        
        Requirements:
        1. Write clean, well-documented code
        2. Include necessary imports
        3. Add error handling where appropriate
        4. Follow best practices for data science
        5. Include comments explaining the logic
        
        Return the code in a code block with proper syntax highlighting.
        """
        return prompt
    
    def _build_analysis_prompt(self, research_question: str, data_context: Dict[str, Any] = None) -> str:
        """Build prompt for task analysis."""
        prompt = f"""
        Analyze this data science research question and provide detailed recommendations:
        
        Research Question: {research_question}
        Data Context: {data_context or 'No data context provided'}
        
        Please provide:
        1. Recommended data sources
        2. Required data preprocessing steps
        3. Appropriate analysis methods
        4. Expected challenges and solutions
        5. Success metrics
        6. Timeline estimates
        
        Format as JSON with detailed recommendations.
        """
        return prompt
    
    async def _async_generate_text(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Generate text using Gemini API with timeout and error handling.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        if not hasattr(self, 'api_key') or not self.api_key:
            self.logger.warning("⚠️ Gemini API key not provided")
            return "API key not available"
        
        try:
            # Generate content with timeout
            response = await asyncio.wait_for(
                self.model.generate_content(prompt),
                timeout=60.0
            )
            
            if response.text:
                return response.text
            else:
                return "No response generated"
                
        except asyncio.TimeoutError:
            self.logger.warning("⚠️ Gemini API request timed out after 60 seconds")
            return "Request timed out - please try again"
            
        except Exception as e:
            error_msg = str(e)
            
            # Handle quota limit errors specifically
            if "429" in error_msg and "quota" in error_msg.lower():
                self.logger.error("🚫 Gemini API quota exceeded (free tier limit: 50 requests/day)")
                self.logger.info("💡 Solutions:")
                self.logger.info("   • Wait for daily quota reset (24 hours)")
                self.logger.info("   • Upgrade to paid Gemini API plan")
                self.logger.info("   • Use fallback methods in the meantime")
                return "API quota exceeded - please wait 24 hours or upgrade your plan"
            
            # Handle other API errors
            elif "403" in error_msg:
                self.logger.error("❌ Gemini API access denied - check your API key")
                return "API access denied - please check your API key"
            elif "404" in error_msg:
                self.logger.error("❌ Gemini model not found - check model configuration")
                return "Model not found - please check configuration"
            else:
                self.logger.error(f"❌ Gemini API error: {error_msg}")
                return f"API error: {error_msg}"
    
    def _extract_code_blocks(self, text: str) -> List[str]:
        """Extract code blocks from generated text."""
        # Look for code blocks marked with ```
        code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', text, re.DOTALL)
        
        # Also look for code blocks without language specification
        if not code_blocks:
            code_blocks = re.findall(r'```\n(.*?)\n```', text, re.DOTALL)
        
        # If no code blocks found, return the entire text
        if not code_blocks:
            return [text.strip()]
        
        return [block.strip() for block in code_blocks]
    
    def _extract_explanation(self, text: str) -> str:
        """Extract explanation from generated text."""
        # Remove code blocks and return the explanation
        explanation = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        return explanation.strip()
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse structured analysis response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback to simple parsing
            return {
                "recommendations": [response],
                "status": "parsed_text"
            }
    
    def _parse_relevance_response(self, response: str) -> Dict[str, Any]:
        """Parse relevance scoring response."""
        # Try to extract score from text
        score_match = re.search(r'score["\s:]*([0-9]*\.?[0-9]+)', response, re.IGNORECASE)
        score = float(score_match.group(1)) if score_match else 0.0
        
        return {
            "relevance_score": score,
            "reasoning": response,
            "status": "parsed_text"
        }
    
    def is_available(self) -> bool:
        """Check if Gemini integration is available."""
        return self.model is not None
