"""
PDF Report Generator for Data Science Agent Swarm

This module generates comprehensive PDF reports from project results,
including all agent outputs, visualizations, and insights.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import pandas as pd


class PDFReportGenerator:
    """
    Generates comprehensive PDF reports from project results.
    
    Features:
    - Executive summary
    - Detailed agent outputs
    - Visualizations and charts
    - Technical appendices
    - Recommendations and insights
    """
    
    def __init__(self, project_id: str, config: Dict[str, Any]):
        """
        Initialize PDF report generator.
        
        Args:
            project_id: Unique project identifier
            config: Configuration settings
        """
        self.project_id = project_id
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup directories
        self.output_dir = Path("output")
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        
        # ReportLab styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        # Heading style
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=12,
            textColor=colors.darkblue
        ))
        
        # Subheading style
        self.styles.add(ParagraphStyle(
            name='CustomSubHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=8,
            spaceBefore=8,
            textColor=colors.darkgreen
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            leading=14
        ))
        
        # Code style
        self.styles.add(ParagraphStyle(
            name='CustomCode',
            parent=self.styles['Code'],
            fontSize=9,
            fontName='Courier',
            leftIndent=20,
            spaceAfter=6
        ))
    
    def generate_project_report(self, project_results: Dict[str, Any]) -> str:
        """
        Generate comprehensive PDF report for a project.
        
        Args:
            project_results: Complete project results dictionary
            
        Returns:
            Path to generated PDF file
        """
        try:
            # Create PDF filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_filename = f"project_report_{self.project_id}_{timestamp}.pdf"
            pdf_path = self.reports_dir / pdf_filename
            
            # Create PDF document
            doc = SimpleDocTemplate(
                str(pdf_path),
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Build story (content)
            story = []
            
            # Add content sections
            story.extend(self._create_title_page(project_results))
            story.append(PageBreak())
            
            story.extend(self._create_executive_summary(project_results))
            story.append(PageBreak())
            
            story.extend(self._create_data_discovery_section(project_results))
            story.append(PageBreak())
            
            story.extend(self._create_analysis_section(project_results))
            story.append(PageBreak())
            
            story.extend(self._create_modeling_section(project_results))
            story.append(PageBreak())
            
            story.extend(self._create_insights_section(project_results))
            story.append(PageBreak())
            
            story.extend(self._create_technical_appendices(project_results))
            
            # Build PDF
            doc.build(story)
            
            self.logger.info(f"PDF report generated: {pdf_path}")
            return str(pdf_path)
            
        except Exception as e:
            self.logger.error(f"Error generating PDF report: {e}")
            raise e
    
    def _create_title_page(self, project_results: Dict[str, Any]) -> List:
        """Create title page."""
        story = []
        
        # Title
        title = Paragraph(
            "Data Science Agent Swarm Project Report",
            self.styles['CustomTitle']
        )
        story.append(title)
        story.append(Spacer(1, 30))
        
        # Project information
        project_info = [
            ["Project ID:", project_results.get('project_id', 'N/A')],
            ["Research Question:", project_results.get('research_question', 'N/A')],
            ["Execution Time:", f"{project_results.get('execution_time', 0):.2f} seconds"],
            ["Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Total Agents:", "12 specialized AI agents"]
        ]
        
        project_table = Table(project_info, colWidths=[2*inch, 4*inch])
        project_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(project_table)
        story.append(Spacer(1, 30))
        
        # System information
        system_info = Paragraph(
            "Generated by Data Science Agent Swarm System<br/>"
            "Powered by Hugging Face Transformers and specialized data science agents",
            self.styles['CustomBody']
        )
        story.append(system_info)
        
        return story
    
    def _create_executive_summary(self, project_results: Dict[str, Any]) -> List:
        """Create executive summary section."""
        story = []
        
        # Section title
        title = Paragraph("Executive Summary", self.styles['CustomHeading'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Research question
        research_q = project_results.get('research_question', 'N/A')
        story.append(Paragraph(f"<b>Research Question:</b> {research_q}", self.styles['CustomBody']))
        story.append(Spacer(1, 12))
        
        # Key findings
        insights = project_results.get('insights', [])
        if insights:
            story.append(Paragraph("<b>Key Findings:</b>", self.styles['CustomSubHeading']))
            for i, insight in enumerate(insights[:5], 1):  # Limit to top 5
                story.append(Paragraph(f"{i}. {insight}", self.styles['CustomBody']))
        else:
            story.append(Paragraph("<b>Key Findings:</b> No insights generated in this run.", self.styles['CustomBody']))
        
        story.append(Spacer(1, 12))
        
        # Recommendations
        recommendations = project_results.get('recommendations', [])
        if recommendations:
            story.append(Paragraph("<b>Recommendations:</b>", self.styles['CustomSubHeading']))
            for i, rec in enumerate(recommendations[:5], 1):  # Limit to top 5
                story.append(Paragraph(f"{i}. {rec}", self.styles['CustomBody']))
        else:
            story.append(Paragraph("<b>Recommendations:</b> No recommendations generated in this run.", self.styles['CustomBody']))
        
        return story
    
    def _create_data_discovery_section(self, project_results: Dict[str, Any]) -> List:
        """Create data discovery section."""
        story = []
        
        title = Paragraph("Data Discovery Results", self.styles['CustomHeading'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        data_discovery = project_results.get('data_discovery', {})
        datasets = data_discovery.get('datasets', [])
        
        if datasets:
            story.append(Paragraph(f"<b>Datasets Found:</b> {len(datasets)}", self.styles['CustomSubHeading']))
            
            for i, dataset in enumerate(datasets[:3], 1):  # Show top 3 datasets
                story.append(Paragraph(f"<b>Dataset {i}:</b>", self.styles['CustomBody']))
                story.append(Paragraph(f"Title: {dataset.get('title', 'N/A')}", self.styles['CustomBody']))
                story.append(Paragraph(f"Description: {dataset.get('description', 'N/A')[:200]}...", self.styles['CustomBody']))
                story.append(Paragraph(f"Relevance Score: {dataset.get('relevance_score', 0):.3f}", self.styles['CustomBody']))
                story.append(Spacer(1, 6))
        else:
            story.append(Paragraph("No datasets were discovered in this run.", self.styles['CustomBody']))
        
        return story
    
    def _create_analysis_section(self, project_results: Dict[str, Any]) -> List:
        """Create analysis section."""
        story = []
        
        title = Paragraph("Data Analysis Results", self.styles['CustomHeading'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        analysis_results = project_results.get('analysis_results', {})
        
        if analysis_results:
            story.append(Paragraph("Analysis was performed on the discovered datasets.", self.styles['CustomBody']))
            
            # Add analysis details if available
            dataset_analyses = analysis_results.get('dataset_analyses', {})
            if dataset_analyses:
                story.append(Paragraph(f"<b>Datasets Analyzed:</b> {len(dataset_analyses)}", self.styles['CustomSubHeading']))
            else:
                story.append(Paragraph("No detailed analysis results available.", self.styles['CustomBody']))
        else:
            story.append(Paragraph("No analysis results available in this run.", self.styles['CustomBody']))
        
        return story
    
    def _create_modeling_section(self, project_results: Dict[str, Any]) -> List:
        """Create modeling section."""
        story = []
        
        title = Paragraph("Modeling Results", self.styles['CustomHeading'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        modeling_results = project_results.get('modeling_results', {})
        models = modeling_results.get('models', [])
        
        if models:
            story.append(Paragraph(f"<b>Models Developed:</b> {len(models)}", self.styles['CustomSubHeading']))
            
            for i, model in enumerate(models[:3], 1):  # Show top 3 models
                story.append(Paragraph(f"<b>Model {i}:</b>", self.styles['CustomBody']))
                story.append(Paragraph(f"Type: {model.get('type', 'N/A')}", self.styles['CustomBody']))
                story.append(Paragraph(f"Performance: {model.get('performance', 'N/A')}", self.styles['CustomBody']))
                story.append(Spacer(1, 6))
        else:
            story.append(Paragraph("No models were developed in this run.", self.styles['CustomBody']))
        
        return story
    
    def _create_insights_section(self, project_results: Dict[str, Any]) -> List:
        """Create insights section."""
        story = []
        
        title = Paragraph("Insights and Recommendations", self.styles['CustomHeading'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        insights = project_results.get('insights', [])
        recommendations = project_results.get('recommendations', [])
        
        if insights:
            story.append(Paragraph("<b>Key Insights:</b>", self.styles['CustomSubHeading']))
            for i, insight in enumerate(insights, 1):
                story.append(Paragraph(f"{i}. {insight}", self.styles['CustomBody']))
            story.append(Spacer(1, 12))
        
        if recommendations:
            story.append(Paragraph("<b>Recommendations:</b>", self.styles['CustomSubHeading']))
            for i, rec in enumerate(recommendations, 1):
                story.append(Paragraph(f"{i}. {rec}", self.styles['CustomBody']))
        else:
            story.append(Paragraph("No insights or recommendations were generated in this run.", self.styles['CustomBody']))
        
        return story
    
    def _create_technical_appendices(self, project_results: Dict[str, Any]) -> List:
        """Create technical appendices."""
        story = []
        
        title = Paragraph("Technical Appendices", self.styles['CustomHeading'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Agent execution details
        story.append(Paragraph("<b>Agent Execution Summary:</b>", self.styles['CustomSubHeading']))
        story.append(Paragraph("The following agents were executed during this project:", self.styles['CustomBody']))
        
        agents = [
            "Dataset Discovery Agent",
            "Data Acquisition Agent", 
            "Data Quality Agent",
            "Feature Engineering Agent",
            "EDA Agent",
            "Statistical Analysis Agent",
            "Model Validation Agent",
            "Visualization Agent",
            "Documentation Agent",
            "Insight Synthesis Agent",
            "Model Architecture Agent",
            "Hyperparameter Optimization Agent"
        ]
        
        for agent in agents:
            story.append(Paragraph(f"• {agent}", self.styles['CustomBody']))
        
        story.append(Spacer(1, 12))
        
        # Configuration details
        story.append(Paragraph("<b>System Configuration:</b>", self.styles['CustomSubHeading']))
        story.append(Paragraph(f"Project ID: {project_results.get('project_id', 'N/A')}", self.styles['CustomBody']))
        story.append(Paragraph(f"Execution Time: {project_results.get('execution_time', 0):.2f} seconds", self.styles['CustomBody']))
        story.append(Paragraph(f"Research Question: {project_results.get('research_question', 'N/A')}", self.styles['CustomBody']))
        
        return story
    
    def _load_agent_outputs(self, agent_name: str) -> Dict[str, Any]:
        """Load outputs from a specific agent."""
        agent_dir = self.output_dir / f"{agent_name}_01"
        if agent_dir.exists():
            for file in agent_dir.glob("*.json"):
                try:
                    with open(file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    self.logger.warning(f"Could not load {file}: {e}")
        return {}
    
    def _create_visualization_section(self) -> List:
        """Create visualization section with charts."""
        story = []
        
        title = Paragraph("Visualizations", self.styles['CustomHeading'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Look for visualization files
        viz_dir = self.output_dir / "visualization_01"
        if viz_dir.exists():
            image_files = list(viz_dir.glob("*.png")) + list(viz_dir.glob("*.jpg"))
            
            for img_file in image_files[:5]:  # Limit to 5 images
                try:
                    img = Image(str(img_file), width=4*inch, height=3*inch)
                    story.append(img)
                    story.append(Spacer(1, 12))
                except Exception as e:
                    self.logger.warning(f"Could not add image {img_file}: {e}")
        else:
            story.append(Paragraph("No visualizations available in this run.", self.styles['CustomBody']))
        
        return story
