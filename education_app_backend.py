from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from flask_sqlalchemy import SQLAlchemy
import jwt
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
from datetime import datetime, timedelta, timezone
import json
import requests
from urllib.parse import urlparse
import sqlite3
from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass
from functools import wraps
import logging
# from docx import Document
import PyPDF2
from bs4 import BeautifulSoup
import time


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///education_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    claude_api_key = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    notes = db.relationship('Note', backref='user', lazy=True, cascade='all, delete-orphan')
    files = db.relationship('File', backref='user', lazy=True, cascade='all, delete-orphan')
    links = db.relationship('Link', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat()
        }

class Note(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    tags = db.Column(db.String(500))  # JSON string of tags
    subject = db.Column(db.String(100))
    priority = db.Column(db.String(20), default='medium')  # low, medium, high
    is_todo = db.Column(db.Boolean, default=False)
    is_completed = db.Column(db.Boolean, default=False)
    due_date = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class File(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(500), nullable=False)
    file_type = db.Column(db.String(50))
    file_size = db.Column(db.Integer)
    description = db.Column(db.Text)
    tags = db.Column(db.String(500))  # JSON string of tags
    subject = db.Column(db.String(100))
    extracted_text = db.Column(db.Text)  # For searchable content
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class Link(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(1000), nullable=False)
    title = db.Column(db.String(200))
    description = db.Column(db.Text)
    favicon = db.Column(db.String(500))
    tags = db.Column(db.String(500))  # JSON string of tags
    subject = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# Helper Classes
@dataclass
class QueryResult:
    content: str
    source_type: str  # 'note', 'file', 'link'
    source_id: int
    title: str
    relevance_score: float
    created_at: datetime

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    relevance_score: float = 0.0

@dataclass
class WebContent:
    url: str
    title: str
    content: str
    summary: str
    metadata: Dict

class ChatSession:
    """Manages individual chat sessions with context and history"""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.chat_history = []
        self.search_context = None
        self.created_at = datetime.now()
    
    def add_message(self, role: str, content: str):
        """Add a message to chat history"""
        self.chat_history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_recent_messages(self, limit: int = 10) -> List[Dict]:
        """Get recent messages for Claude API context"""
        recent_messages = []
        for msg in self.chat_history[-limit:]:
            if msg['role'] in ['user', 'assistant']:
                recent_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        return recent_messages
    
    def set_search_context(self, context: List):
        """Set or update search context"""
        self.search_context = context



class DuckDuckGoSearcher:
    """Enhanced DuckDuckGo search with rate limiting and error handling"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.last_request_time = 0
        self.rate_limit_delay = 1.5  # Seconds between requests
    
    def _rate_limit(self):
        """Ensure rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search DuckDuckGo for results"""
        self._rate_limit()
        
        try:
            # DuckDuckGo Instant Answer API
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = self.session.get('https://api.duckduckgo.com/', params=params)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Parse instant answer results
            if data.get('Results'):
                for result in data['Results'][:max_results]:
                    results.append(SearchResult(
                        title=result.get('Text', ''),
                        url=result.get('FirstURL', ''),
                        snippet=result.get('Result', '')
                    ))
            
            # If no instant answers, try HTML search
            if not results:
                results = self._html_search(query, max_results)
            
            return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []
    
    def _html_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Fallback HTML search when API doesn't return results"""
        try:
            params = {'q': query, 'format': 'html'}
            response = self.session.get('https://duckduckgo.com/html/', params=params)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Parse search results from HTML
            for result_div in soup.find_all('div', class_='result')[:max_results]:
                title_elem = result_div.find('a', class_='result__a')
                snippet_elem = result_div.find('a', class_='result__snippet')
                
                if title_elem:
                    results.append(SearchResult(
                        title=title_elem.get_text(strip=True),
                        url=title_elem.get('href', ''),
                        snippet=snippet_elem.get_text(strip=True) if snippet_elem else ''
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"HTML search error: {e}")
            return []

class WebContentExtractor:
    """Extract and clean content from web pages"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def extract_content(self, url: str) -> Optional[WebContent]:
        """Extract content from a web page"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            # Extract title
            title = ''
            if soup.title:
                title = soup.title.string.strip()
            elif soup.find('h1'):
                title = soup.find('h1').get_text(strip=True)
            
            # Extract main content
            content_selectors = [
                'article', 'main', '[role="main"]', 
                '.content', '.post-content', '.entry-content',
                '.article-content', '.page-content'
            ]
            
            content = ''
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content = content_elem.get_text(separator='\n', strip=True)
                    break
            
            # Fallback to body if no main content found
            if not content and soup.body:
                content = soup.body.get_text(separator='\n', strip=True)
            
            # Clean up content
            content = self._clean_text(content)
            
            # Extract metadata
            metadata = {
                'description': self._get_meta_content(soup, 'description'),
                'keywords': self._get_meta_content(soup, 'keywords'),
                'author': self._get_meta_content(soup, 'author'),
                'og_title': self._get_meta_content(soup, 'og:title'),
                'og_description': self._get_meta_content(soup, 'og:description'),
            }
            
            return WebContent(
                url=url,
                title=title,
                content=content,
                summary='',  # Will be generated by Claude
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Content extraction error for {url}: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:  # Filter out very short lines
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _get_meta_content(self, soup: BeautifulSoup, name: str) -> str:
        """Extract meta tag content"""
        meta = soup.find('meta', {'name': name}) or soup.find('meta', {'property': name})
        return meta.get('content', '') if meta else ''

class RAGSystem:
    def __init__(self):
        self.claude_api_url = "https://api.anthropic.com/v1/messages"
        self.chat_sessions = {}  # Store active chat sessions
        self.searcher = DuckDuckGoSearcher()
        self.content_extractor = WebContentExtractor()

    def search_content(self, api_key, user_id: int, query: str, content_types: List[str] = None) -> List[QueryResult]:
        """Search through user's content using simple text matching"""
        results = []
        
        if not content_types:
            content_types = ['notes', 'files', 'links']
        
        # Search notes
        if 'notes' in content_types:
            notes = Note.query.filter_by(user_id=user_id).all()
            for note in notes:
                searchable_text = note.title + " " + note.content
                # score = self._calculate_relevance(query, note.title + " " + note.content)
                score = self.calculate_relevance_with_claude(query, searchable_text, api_key)
                if score > 0.1:  # Threshold for relevance
                    results.append(QueryResult(
                        content=note.content,
                        source_type='note',
                        source_id=note.id,
                        title=note.title,
                        relevance_score=score,
                        created_at=note.created_at
                    ))
        
        # Search files
        if 'files' in content_types:
            files = File.query.filter_by(user_id=user_id).all()
            for file in files:
                searchable_text = f"{file.original_filename} {file.description} {file.extracted_text}"
                score = self.calculate_relevance_with_claude(query, searchable_text, api_key)
                if score > 0.1:
                    results.append(QueryResult(
                        content=file.description,
                        source_type='file',
                        source_id=file.id,
                        title=file.original_filename,
                        relevance_score=score,
                        created_at=file.created_at
                    ))
        
        # Search links
        if 'links' in content_types:
            links = Link.query.filter_by(user_id=user_id).all()
            for link in links:
                searchable_text = f"{link.title} {link.description} {link.url}"
                # score = self._calculate_relevance(query, searchable_text)
                score = self.calculate_relevance_with_claude(query, searchable_text, api_key)
                if score > 0.1:
                    results.append(QueryResult(
                        content=link.description or link.title or link.url,
                        source_type='link',
                        source_id=link.id,
                        title=link.title or link.url,
                        relevance_score=score,
                        created_at=link.created_at
                    ))
        
        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:10]  # Return top 10 results
    
    def calculate_relevance_with_claude(self, query: str, content: str, api_key: str) -> float:
        """Use Claude to calculate semantic relevance"""
        if not content or not query:
            return 0.0
        
        # Truncate content if too long (Claude has token limits)
        max_content_length = 8000  # Adjust based on your needs
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        
        prompt = f"""
        Rate the relevance of the following document content to the user's query on a scale of 0.0 to 1.0, where:
        - 0.0 = completely irrelevant
        - 0.5 = somewhat relevant
        - 1.0 = highly relevant
        
        Consider semantic meaning, not just keyword matching. Look for conceptual relationships, synonyms, and contextual relevance.
        
        User Query: "{query}"
        
        Document Content: "{content}"
        
        Respond with ONLY a decimal number between 0.0 and 1.0, nothing else.
        """
        
        try:
            response = self.query_claude_simple(api_key, prompt)
            # print("response: ", response)
            # Extract just the number from Claude's response
            score = float(response.get("response"))
            return max(0.0, min(1.0, score))  # Ensure it's between 0 and 1
        except Exception as e:
            print(f"Error calculating relevance with Claude: {e}")
            # Fallback to simple method
            return self._calculate_relevance(query, content)
    
    def _calculate_relevance(self, query: str, content: str) -> float:
        """Simple relevance scoring based on keyword matching"""
        if not content:
            return 0.0
        
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        # Calculate Jaccard similarity
        intersection = query_words.intersection(content_words)
        union = query_words.union(content_words)
        
        if len(union) == 0:
            return 0.0
        
        return len(intersection) / len(union)
    
    
    def query_claude_b(self, api_key: str, query: str, context: List[QueryResult], 
                    session_id: Optional[str] = None) -> Dict:
        """Query Claude API with context from user's data and chat history"""
        if not api_key:
            return {
                "response": "Claude API key not configured. Please add your API key in settings.",
                "session_id": session_id,
                "error": True
            }
        
        # Generate session ID if not provided
        if not session_id:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(query)}"
        
        # Get or create chat session
        if session_id not in self.chat_sessions:
            self.chat_sessions[session_id] = ChatSession(session_id)
        
        session = self.chat_sessions[session_id]
        
              
        # Add user message to history
        session.add_message('user', query)
        
        # Prepare context text from search results
        context_text = self._prepare_context_text(context.get("results", []))
        
        # Get recent chat history
        recent_messages = session.get_recent_messages()
        
        # Build messages for Claude API
        messages = self._build_claude_messages(query, context_text, recent_messages)
        
        # Prepare API request
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1500,
            "messages": messages
        }
        
        try:
            response = requests.post(self.claude_api_url, headers=headers, json=data)
            response.raise_for_status()
            
            claude_response = response.json()["content"][0]["text"]
            
            # Add Claude's response to history
            session.add_message('assistant', claude_response)
            
            return {
                "response": claude_response,
                "session_id": session_id,
                "message_count": len(session.chat_history),
                "error": False
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Claude API error: {e}")
            error_msg = f"Error querying Claude: {str(e)}"
            session.add_message('system', f"Error: {error_msg}")
            return {
                "response": error_msg,
                "session_id": session_id,
                "error": True
            }
        except KeyError as e:
            logger.error(f"Unexpected Claude API response format: {e}")
            error_msg = "Error: Unexpected response format from Claude API"
            session.add_message('system', f"Error: {error_msg}")
            return {
                "response": error_msg,
                "session_id": session_id,
                "error": True
            }

    def query_claude_simple(self, api_key: str, query: str) -> Dict:
        """Query Claude API with context from user's data and chat history"""
        if not api_key:
            return {
                "response": "Claude API key not configured. Please add your API key in settings.",
                "error": True
            }
        
        # Build messages for Claude API
        messages = self._build_claude_simple_msg(query)
        
        # Prepare API request
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1500,
            "messages": messages
        }
        
        try:
            response = requests.post(self.claude_api_url, headers=headers, json=data)
            response.raise_for_status()
            
            claude_response = response.json()["content"][0]["text"]
            
            return {
                "response": claude_response,
                "error": False
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Claude API error: {e}")
            error_msg = f"Error querying Claude: {str(e)}"
            return {
                "response": error_msg,
                "error": True
            }
        except KeyError as e:
            logger.error(f"Unexpected Claude API response format: {e}")
            error_msg = "Error: Unexpected response format from Claude API"
            return {
                "response": error_msg,
                "error": True
            }
        
    def query_claude_base(self, api_key: str, query: str) -> Dict:
        """Query Claude API with context from user's data and chat history"""
        if not api_key:
            return {
                "response": "Claude API key not configured. Please add your API key in settings.",
                "error": True
            }
        
        # Build messages for Claude API
        messages = self._build_claude_simple_msg(query)
        
        # Prepare API request
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1500,
            "messages": messages
        }
        
        try:
            response = requests.post(self.claude_api_url, headers=headers, json=data)
            response.raise_for_status()
            
            claude_response = response.json()["content"][0]["text"]
            
            return {
                "response": claude_response,
                "error": False
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Claude API error: {e}")
            error_msg = f"Error querying Claude: {str(e)}"
            return {
                "response": error_msg,
                "error": True
            }
        except KeyError as e:
            logger.error(f"Unexpected Claude API response format: {e}")
            error_msg = "Error: Unexpected response format from Claude API"
            return {
                "response": error_msg,
                "error": True
            }


    def query_claude_files(self, api_key: str, query: str, context: str, 
                    session_id: Optional[str] = None) -> Dict:
        """Query Claude API with context from user's data and chat history"""
        if not api_key:
            return {
                "response": "Claude API key not configured. Please add your API key in settings.",
                "session_id": session_id,
                "error": True
            }
        
        # Generate session ID if not provided
        if not session_id:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(query)}"
        
        # Get or create chat session
        if session_id not in self.chat_sessions:
            self.chat_sessions[session_id] = ChatSession(session_id)
        
        session = self.chat_sessions[session_id]
        
        # Add user message to history
        session.add_message('user', query)
        
        # Prepare context text from search results
        context_text = context
        # Get recent chat history
        recent_messages = session.get_recent_messages()
        
        # Build messages for Claude API
        messages = self._build_claude_messages(query, context_text, recent_messages)
        
        # Prepare API request
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1500,
            "messages": messages
        }
        
        try:
            response = requests.post(self.claude_api_url, headers=headers, json=data)
            response.raise_for_status()
            
            claude_response = response.json()["content"][0]["text"]
            
            # Add Claude's response to history
            session.add_message('assistant', claude_response)
            
            return {
                "response": claude_response,
                "session_id": session_id,
                "message_count": len(session.chat_history),
                "error": False
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Claude API error: {e}")
            error_msg = f"Error querying Claude: {str(e)}"
            session.add_message('system', f"Error: {error_msg}")
            return {
                "response": error_msg,
                "session_id": session_id,
                "error": True
            }
        except KeyError as e:
            logger.error(f"Unexpected Claude API response format: {e}")
            error_msg = "Error: Unexpected response format from Claude API"
            session.add_message('system', f"Error: {error_msg}")
            return {
                "response": error_msg,
                "session_id": session_id,
                "error": True
            }
    
    def _prepare_context_text(self, context: List[QueryResult]) -> str:
        """Prepare context text from search results"""
        if not context:
            return ""
        
        context_text = ""
        for result in context:
            source_type = result.get("source_type").title()
            title = result.get("title")
            context_text += f"\n--- {source_type}: {title} ---\n"
            # Truncate content to avoid token limits
            content = result.get("content")
            if len(content) > 1000:  # Limit content length
                content = content[:1000] + "..."
            context_text += f"{content}\n"
        
        return context_text
    
    def _build_claude_messages(self, query: str, context_text: str, 
                              recent_messages: List[Dict]) -> List[Dict]:
        """Build messages array for Claude API"""
        
        # If this is a new conversation, include full context
        if not recent_messages:
            system_prompt = f"""You are an AI assistant helping a student with their educational content.

                Here is the student's relevant content from their notes, files, and links:
                {context_text}

                Student's question: {query}

                Please provide a helpful response based on the student's content. If you're summarizing or answering about specific dates, focus on the most relevant information."""
            return [{"role": "user", "content": system_prompt}]
        
        # For continuing conversations, use chat history
        messages = []
        
        # Add context as first message if not already included
        if len(recent_messages) <= 2:  # Only add context for early messages
            context_msg = f"""Previous search context:
                {context_text}

                Continue our conversation about the student's educational content."""
            messages.append({"role": "user", "content": context_msg})
        
        # Add recent chat history
        messages.extend(recent_messages[:-1])  # Exclude the current user message
        
        # Add current user message
        messages.append({"role": "user", "content": query})
        
        return messages
    
    def _build_claude_simple_msg(self, query: str):
            """Build messages array for Claude API"""
            
            # If this is a new conversation, include full context
            system_prompt = f"""{query}"""
            return [{"role": "user", "content": system_prompt}]
    
    def get_chat_history(self, session_id: str) -> Dict:
        """Get chat history for a session"""
        if session_id not in self.chat_sessions:
            return {"history": [], "search_context": None}
        
        session = self.chat_sessions[session_id]
        return {
            "history": session.chat_history,
            "search_context": session.search_context,
            "session_id": session_id,
            "created_at": session.created_at.isoformat()
        }
    
    def clear_chat_session(self, session_id: str) -> bool:
        """Clear a specific chat session"""
        if session_id in self.chat_sessions:
            del self.chat_sessions[session_id]
            return True
        return False
    
    def list_active_sessions(self) -> List[Dict]:
        """List all active chat sessions"""
        sessions = []
        for session_id, session in self.chat_sessions.items():
            sessions.append({
                "session_id": session_id,
                "created_at": session.created_at.isoformat(),
                "message_count": len(session.chat_history),
                "has_search_context": session.search_context is not None
            })
        return sessions
    
    def save_chat_session(self, session_id: str, filename: Optional[str] = None) -> str:
        """Save chat session to file"""
        if session_id not in self.chat_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.chat_sessions[session_id]
        
        if not filename:
            filename = f"chat_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        save_data = {
            "session_id": session_id,
            "created_at": session.created_at.isoformat(),
            "saved_at": datetime.now().isoformat(),
            "search_context": [
                {
                    "source_type": result.source_type,
                    "title": result.title,
                    "content": result.content
                } for result in (session.search_context or [])
            ],
            "chat_history": session.chat_history
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        return filename
    
    def continue_chat(self, api_key: str, query: str, session_id: str) -> Dict:
        """Continue an existing chat session"""
        if session_id not in self.chat_sessions:
            return {
                "response": "Chat session not found. Please start a new conversation.",
                "session_id": session_id,
                "error": True
            }
        
        # Use existing session's search context
        session = self.chat_sessions[session_id]
        return self.query_claude(api_key, query, session.search_context or [], session_id)
    
    def query_claude(self, api_key: str, query: str, context: List[QueryResult]) -> str:
        """Query Claude API with context from user's data"""
        if not api_key:
            return "Claude API key not configured. Please add your API key in settings."
        
        # Prepare context for Claude
        context_text = ""
        for result in context:
            context_text += f"\n--- {result.source_type.title()}: {result.title} ---\n"
            context_text += f"{result.content}\n"
        
        if context_text == "":
            messages = [
                {
                    "role": "user",
                    "content": f"""You are an AI assistant helping a student with their educational content. 
                    Student's question: {query}

                    Please provide a helpful response based on the student's content. If you're summarizing or answering about specific dates, focus on the most relevant information."""
                }
            ]
        else:   
            # Prepare the message for Claude
            messages = [
                {
                    "role": "user",
                    "content": f"""You are an AI assistant helping a student with their educational content. 
                    
                    Here is the student's relevant content from their notes, files, and links:
                    {context_text}

                    Student's question: {query}

                    Please provide a helpful response based on the student's content. If you're summarizing or answering about specific dates, focus on the most relevant information."""
                }
            ]
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": messages
        }
        
        try:
            response = requests.post(self.claude_api_url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()["content"][0]["text"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Claude API error: {e}")
            return f"Error querying Claude: {str(e)}"
        except KeyError as e:
            logger.error(f"Unexpected Claude API response format: {e}")
            return "Error: Unexpected response format from Claude API"

    def claude_with_search(self, api_key: str, query: str, urls: List[str] = None, 
                          search_enabled: bool = True) -> Dict:
        """
        Enhanced Claude integration with DuckDuckGo search and URL summarization
        
        Args:
            api_key: Claude API key
            query: User's query
            urls: Optional list of URLs to summarize
            search_enabled: Whether to perform web search
        """
        if not api_key:
            return {
                "response": "Claude API key not configured. Please add your API key in settings.",
                "error": True
            }
        
        try:
            # Step 1: Extract content from provided URLs
            url_contents = []
            if urls:
                for url in urls:
                    content = self.content_extractor.extract_content(url)
                    if content:
                        url_contents.append(content)
            
            # Step 2: Generate search query if search is enabled
            search_results = []
            if search_enabled:
                search_query = self._generate_search_query(api_key, query, url_contents)
                if search_query:
                    search_results = self.searcher.search(search_query, max_results=5)
            
            # Step 3: Extract content from top search results
            search_contents = []
            for result in search_results[:3]:  # Limit to top 3 results
                if result.url:
                    content = self.content_extractor.extract_content(result.url)
                    if content:
                        search_contents.append(content)
            
            # Step 4: Generate comprehensive response
            response = self._generate_comprehensive_response(
                api_key, query, url_contents, search_contents, search_results
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Claude with search error: {e}")
            return {
                "response": f"Error processing request: {str(e)}",
                "error": True
            }
    
    def _generate_search_query(self, api_key: str, user_query: str, 
                              url_contents: List[WebContent]) -> Optional[str]:
        """Generate an optimal search query using Claude"""
        
        context = ""
        if url_contents:
            context = "\n\n".join([
                f"Content from {content.url}:\nTitle: {content.title}\nSummary: {content.content[:500]}..."
                for content in url_contents
            ])
        
        prompt = f"""Based on the user's query and any provided content, generate a focused search query for DuckDuckGo that would find the most relevant additional information.

            User Query: {user_query}

            Context from provided URLs:
            {context}

            Please generate a concise search query (2-6 words) that would help find relevant information to better answer the user's question. Only return the search query, nothing else."""

        try:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            response = requests.post(self.claude_api_url, headers=headers, json=data)
            response.raise_for_status()
            
            search_query = response.json()["content"][0]["text"].strip()
            logger.info(f"Generated search query: {search_query}")
            return search_query
            
        except Exception as e:
            logger.error(f"Error generating search query: {e}")
            return None
    
    def _generate_comprehensive_response(self, api_key: str, user_query: str,
                                       url_contents: List[WebContent],
                                       search_contents: List[WebContent],
                                       search_results: List[SearchResult]) -> Dict:
        """Generate comprehensive response using all available information"""
        
        # Build context from URL contents
        url_context = ""
        if url_contents:
            url_context = "\n\n".join([
                f"=== Content from {content.url} ===\nTitle: {content.title}\nContent: {content.content[:2000]}..."
                for content in url_contents
            ])
        
        # Build context from search results
        search_context = ""
        if search_contents:
            search_context = "\n\n".join([
                f"=== Search Result: {content.title} ===\nURL: {content.url}\nContent: {content.content[:1500]}..."
                for content in search_contents
            ])
        
        # Build search results summary
        results_summary = ""
        if search_results:
            results_summary = "\n".join([
                f"- {result.title}: {result.snippet}"
                for result in search_results[:5]
            ])
        
        prompt = f"""You are an AI assistant that provides comprehensive answers by analyzing provided content and web search results.

            User Query: {user_query}

            Provided URL Content:
            {url_context}

            Related Web Search Results:
            {search_context}

            Search Results Summary:
            {results_summary}

            Instructions:
            1. Analyze the user's query carefully
            2. If URLs were provided, summarize their key points relevant to the query
            3. Use the search results to provide additional context and up-to-date information
            4. Synthesize all information into a comprehensive, well-structured response
            5. Cite sources when referencing specific information
            6. If there are any contradictions between sources, mention them
            7. Provide actionable insights when possible

            Please provide a thorough response that addresses the user's query using all available information."""

        try:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 2000,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            response = requests.post(self.claude_api_url, headers=headers, json=data)
            response.raise_for_status()
            
            claude_response = response.json()["content"][0]["text"]
            
            # Add metadata about sources used
            sources_used = []
            if url_contents:
                sources_used.extend([content.url for content in url_contents])
            if search_contents:
                sources_used.extend([content.url for content in search_contents])
            
            return {
                "response": claude_response,
                "sources_used": sources_used,
                "search_query_used": len(search_results) > 0,
                "num_sources": len(sources_used),
                "error": False
            }
            
        except Exception as e:
            logger.error(f"Error generating comprehensive response: {e}")
            return {
                "response": f"Error generating response: {str(e)}",
                "error": True
            }
    
    def summarize_url(self, api_key: str, url: str, focus_query: str = None) -> Dict:
        """Summarize a specific URL with optional focus"""
        
        # Extract content
        content = self.content_extractor.extract_content(url)
        if not content:
            return {
                "response": f"Unable to extract content from {url}",
                "error": True
            }
        
        # Build prompt
        if focus_query:
            prompt = f"""Please analyze and summarize the following web page content, focusing on: {focus_query}
                URL: {url}
                Title: {content.title}

                Content:
                {content.content[:3000]}

                Please provide:
                1. A comprehensive summary of the content
                2. Key points relevant to: {focus_query}
                3. Any important insights or takeaways
                4. Relevant quotes or data points"""
        else:
            prompt = f"""Please analyze and summarize the following web page content:
                URL: {url}
                Title: {content.title}

                Content:
                {content.content[:3000]}

                Please provide:
                1. A comprehensive summary of the main points
                2. Key insights and takeaways
                3. Important data or quotes
                4. The overall purpose/message of the content"""
        
        try:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1500,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            response = requests.post(self.claude_api_url, headers=headers, json=data)
            response.raise_for_status()
            
            claude_response = response.json()["content"][0]["text"]
            
            return {
                "response": claude_response,
                "url": url,
                "title": content.title,
                "metadata": content.metadata,
                "error": False
            }
            
        except Exception as e:
            logger.error(f"Error summarizing URL {url}: {e}")
            return {
                "response": f"Error summarizing URL: {str(e)}",
                "error": True
            }



# Initialize RAG system
rag_system = RAGSystem()



@app.route('/api/claude-chat', methods=['POST'])
def claude_chat():
    try:
        data = request.get_json()
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'message': 'No token provided'}), 401
        
        # Handle "Bearer <token>" format
        if token.startswith('Bearer '):
            token = token.split(' ')[1]
        
        user_id = verify_token_b(token)
        if not user_id:
            return jsonify({'message': 'Invalid or expired token'}), 401
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        message = data.get('message', '').strip()
        context = data.get('context', {})
        session_id = data.get('session_id', 'default')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Get or create chat session
        if session_id not in rag_system.chat_sessions:
            rag_system.chat_sessions[session_id] = ChatSession(session_id)
        
        session = rag_system.chat_sessions[session_id]
        
        # # Update search context if provided
        # if context:
        #     session.search_context = context
        
        # # Add user message to history
        # session.add_message('user', message)
        
        
        # Build system prompt with search context
        # system_prompt = build_system_prompt(context)
        
        user = User.query.get(user_id)
        if not user.claude_api_key:
            return jsonify({'error': 'Claude API key not configured'}), 400
        # Call Claude API
        response = rag_system.query_claude_b(
            user.claude_api_key, message, context, session_id
        )
        
        claude_response = response.get("response")
        
        # Add Claude's response to history
        session.add_message('assistant', claude_response)
        
        return jsonify({
            'response': claude_response,
            'session_id': session_id,
            'message_count': len(session.chat_history)
        })
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

def build_system_prompt(search_context, message):
    """Build system prompt with search context"""
    if not search_context:
        return "You are a helpful assistant. Please provide clear and informative responses."
    
    query = search_context.get('query', '')
    results = search_context.get('results', [])
    
    context_summary = ""
    if results:
        context_summary = "\n\nSearch Results:\n"
        for i, result in enumerate(results[:5]):  # Limit to top 5 results
            context_summary += f"{i+1}. {result.get('type', 'item').title()}: {result.get('title', 'Untitled')}\n"
            content = result.get('content', '')
            # Truncate content to avoid token limits
            if len(content) > 200:
                content = content[:200] + "..."
            context_summary += f"   Content: {content}\n"
            if result.get('tags'):
                context_summary += f"   Tags: {', '.join(result['tags'])}\n"
            context_summary += "\n"
    
    system_prompt = f"""You are a helpful assistant analyzing search results and having a conversation about them.

        Original search query: "{query}"
        {context_summary}

        Please provide helpful responses based on the search context and conversation history. When referencing specific results, you can refer to them by their titles or types. Be conversational and helpful."""

    return system_prompt

@app.route('/api/chat-history/<session_id>', methods=['GET'])
def get_chat_history(session_id):
    """Get chat history for a session"""
    try:
        if session_id not in rag_system.chat_sessions:
            return jsonify({'history': []}), 200
        
        session = rag_system.chat_sessions[session_id]
        return jsonify({
            'history': session.chat_history,
            'search_context': session.search_context
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/clear-chat/<session_id>', methods=['POST'])
def clear_chat(session_id):
    """Clear chat history for a session"""
    try:
        if session_id in rag_system.chat_sessions:
            del rag_system.chat_sessions[session_id]
        
        return jsonify({'message': 'Chat history cleared'}), 200
        
    except Exception as e:
        logger.error(f"Error clearing chat: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/save-chat/<session_id>', methods=['POST'])
def save_chat(session_id):
    """Save chat session to file (or database)"""
    try:
        if session_id not in rag_system.chat_sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session = rag_system.chat_sessions[session_id]
        
        # Create saves directory if it doesn't exist
        os.makedirs('saved_chats', exist_ok=True)
        
        # Save to JSON file
        filename = f"saved_chats/chat_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        save_data = {
            'session_id': session_id,
            'created_at': session.created_at.isoformat(),
            'saved_at': datetime.now().isoformat(),
            'search_context': session.search_context,
            'chat_history': session.chat_history
        }
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        return jsonify({
            'message': 'Chat saved successfully',
            'filename': filename
        }), 200
        
    except Exception as e:
        logger.error(f"Error saving chat: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_sessions': len(rag_system.chat_sessions)
    }), 200

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """List active chat sessions"""
    try:
        sessions_info = []
        for session_id, session in rag_system.chat_sessions.items():
            sessions_info.append({
                'session_id': session_id,
                'created_at': session.created_at.isoformat(),
                'message_count': len(session.chat_history),
                'has_search_context': session.search_context is not None
            })
        
        return jsonify({'sessions': sessions_info}), 200
        
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# Helper Functions
def allowed_file(filename):
    # ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx', 'ppt', 'pptx', 'xls', 'xlsx'}
    ALLOWED_EXTENSIONS = {'txt', 'pdf'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_metadata_from_url(url):
    """Extract metadata from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Simple title extraction
        title_match = re.search(r'<title>(.*?)</title>', response.text, re.IGNORECASE)
        title = title_match.group(1) if title_match else url
        
        # Simple description extraction
        desc_match = re.search(r'<meta name="description" content="(.*?)"', response.text, re.IGNORECASE)
        description = desc_match.group(1) if desc_match else ""
        
        return {
            'title': title.strip(),
            'description': description.strip(),
            'favicon': f"https://www.google.com/s2/favicons?domain={urlparse(url).netloc}"
        }
    except Exception as e:
        logger.error(f"Error extracting metadata from {url}: {e}")
        return {
            'title': url,
            'description': "",
            'favicon': ""
        }

# Authentication decorator
def requires_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'message': 'No token provided'}), 401
        
        # Handle "Bearer <token>" format
        if token.startswith('Bearer '):
            token = token.split(' ')[1]
        
        user_id = verify_token_b(token)
        if not user_id:
            return jsonify({'message': 'Invalid or expired token'}), 401
        
        # Get user and attach to request
        user = User.query.get(user_id)
        if not user:
            return jsonify({'message': 'User not found'}), 401
        
        request.current_user = user
        return f(*args, **kwargs)
    
    return decorated_function

# CORS handling (if needed)
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('rag_frontend'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def logina():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            return jsonify({'success': True, 'redirect': url_for('personalArchiveHomepage')})
        else:
            return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def registera():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            return jsonify({'success': False, 'error': 'Username already exists'}), 400
        
        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'error': 'Email already exists'}), 400
        
        # Create new user
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        
        db.session.add(user)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'User created successfully'})
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/archive')
# @requires_auth
def archive():
    # user = User.query.get(session['user_id'])
    return render_template('personalArchiveHomepage.html')

# API Routes
@app.route('/api/notesb', methods=['GET', 'POST'])
#@requires_auth
def handle_notesb():
    token = request.headers.get('Authorization')
    if token and token.startswith('Bearer '):
        token = token.split(' ')[1]
        user_id = verify_token_b(token)
    # Get user and validate API key
    user = User.query.get(user_id)
    
    if not user:
        return {'error': 'User not found'}, 404
    
    if not user.claude_api_key:
        return {'error': 'Claude API key not configured'}, 400
    if request.method == 'GET':
        
        notes = Note.query.filter_by(user_id=user_id).order_by(Note.created_at.desc()).all()
        return jsonify([{
            'id': note.id,
            'title': note.title,
            'content': note.content,
            'tags': json.loads(note.tags) if note.tags else [],
            'subject': note.subject,
            'priority': note.priority,
            'is_todo': note.is_todo,
            'is_completed': note.is_completed,
            'due_date': note.due_date.isoformat() if note.due_date else None,
            'created_at': note.created_at.isoformat()
        } for note in notes])
    
    elif request.method == 'POST':
        data = request.get_json()
        message = f"""Summarize the contents of the document context for my note page. Return only the results needed for the note.
            Remove any characters, just display some text from each source, if any. Add the entire chat context, if any. 
            {data}
        """
        
        session_id = 'default'
        context = ""
        
        result =  rag_system.query_claude_simple(user.claude_api_key, message)
        # Update the description with Claude's response
        if result:
            # Extract just the response text, not the whole result dict
            claude_response = result.get('response', '')
            note = Note(
                title=data.get('title'),
                content=claude_response,
                tags=json.dumps(data.get('tags', [])),
                subject=data.get('subject'),
                priority=data.get('priority', 'medium'),
                is_todo=data.get('is_todo', False),
                due_date=datetime.fromisoformat(data['due_date']) if data.get('due_date') else None,
                user_id=user_id
            )
            
            db.session.add(note)
            db.session.commit()
            
            return jsonify({'success': True, 'id': note.id})
            
        else:
            # Handle error case
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error')
            }), status_code
       

@app.route('/api/notes/<int:note_id>', methods=['PUT','DELETE'])
# @requires_auth
def handle_note(note_id):
    token = request.headers.get('Authorization')
    if token and token.startswith('Bearer '):
        token = token.split(' ')[1]
        user_id = verify_token_b(token)
    note = Note.query.filter_by(id=note_id, user_id=user_id).first()
    
    if not note:
        return jsonify({'error': 'Note not found'}), 404
    if request.method == 'PUT':
        data = request.get_json()
        note.title = data.get('title', note.title)
        note.content = data.get('content', note.content)
        note.tags = json.dumps(data.get('tags', json.loads(note.tags) if note.tags else []))
        note.subject = data.get('subject', note.subject)
        note.priority = data.get('priority', note.priority)
        note.is_todo = data.get('is_todo', note.is_todo)
        note.is_completed = data.get('is_completed', note.is_completed)
        note.due_date = datetime.fromisoformat(data['due_date']) if data.get('due_date') else note.due_date
        note.updated_at = datetime.utcnow()
        
        db.session.commit()
        return jsonify({'success': True})
    
    elif request.method == 'DELETE':
        db.session.delete(note)
        db.session.commit()
        return jsonify({'success': True})

def call_claude_with_file(user_id, message, file_id, session_id, context=""):
    """
    Helper function to call Claude API with file contents included
    
    Args:
        user_id: ID of the user making the request
        message: The user's message/query
        file_id: ID of the file to include in the context
        session_id: Session ID for conversation history
        context: Additional context (optional)
    
    Returns:
        dict: Response from Claude API or error message
    """
    try:
        # Get user and validate API key
        user = User.query.get(user_id)
        
        if not user:
            return {'error': 'User not found'}, 404
        
        if not user.claude_api_key:
            return {'error': 'Claude API key not configured'}, 400
        
        # Get file from database
        file_record = File.query.get(file_id)  # Adjust based on your File 
        
        if not file_record:
            return {'error': 'File not found'}, 404
        
        # Read file contents
        file_path = os.path.join('uploads', file_record.filename)  # Adjust path as needed
        if not os.path.exists(file_path):
            return {'error': 'File not found on disk'}, 404
        
        # Read file content based on file type
        file_content = ""
        file_extension = os.path.splitext(file_record.filename)[1].lower()
        
        if file_extension in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json']:
            # Text files
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
        elif file_extension in ['.pdf']:
            # PDF files (requires PyPDF2 or similar)
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                file_content = ""
                for page in reader.pages:
                    file_content += page.extract_text()
        # elif file_extension in ['.docx']:
            # Word documents (requires python-docx)
            
            # doc = Document(file_path)
            # file_content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        else:
            return {'error': f'Unsupported file type: {file_extension}'}, 400
        
        # Prepare enhanced context with file contents
        enhanced_context = f"""
            {context}

            File: {file_record.filename}
            Content:
            {file_content}
            """
        
        #print("got prep: ", enhanced_context)
        
        # Call Claude API with file contents
        response = rag_system.query_claude_files(
            user.claude_api_key, 
            message, 
            enhanced_context, 
            session_id
        )
        
        # print("response: ", type(response))
        claude_response = response.get("response")
        
        # Add Claude's response to history
        # session.add_message('assistant', claude_response)
        
        return {
            'response': claude_response,
            'file_included': file_record.filename
        }, 200
        
    except Exception as e:
        print(f"Error in call_claude_with_file: {str(e)}")
        return {'error': 'Internal server error'}, 500

@app.route('/api/files', methods=['GET', 'POST'])
# @requires_auth
def handle_files():
    token = request.headers.get('Authorization')
    if token and token.startswith('Bearer '):
        token = token.split(' ')[1]
        user_id = verify_token_b(token)
    if request.method == 'GET':
        files = File.query.filter_by(user_id=user_id).order_by(File.created_at.desc()).all()
        return jsonify([{
            'id': file.id,
            'filename': file.original_filename,
            'file_type': file.file_type,
            'file_size': file.file_size,
            'description': file.description,
            'tags': json.loads(file.tags) if file.tags else [],
            'subject': file.subject,
            'created_at': file.created_at.isoformat()
        } for file in files])
    
    elif request.method == 'POST':
        if 'files' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['files']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Add timestamp to avoid conflicts
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_')
            filename = timestamp + filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Create file record
            file_record = File(
                filename=filename,
                original_filename=file.filename,
                filepath=filepath,
                file_type=file.filename.rsplit('.', 1)[1].lower(),
                file_size=os.path.getsize(filepath),
                description=request.form.get('description', ''),
                tags=json.dumps(request.form.get('tags', '').split(',') if request.form.get('tags') else []),
                subject=request.form.get('subject', ''),
                user_id=user_id
            )
            db.session.add(file_record)
            db.session.commit()
            
            available_files = ["txt", "pdf", '.md', 'py', 'js', 'html', 'css', 'json']
            if file_record.file_type in available_files:
                message = "Summarize the contents of the document context provided for academic purposes. I am a student continually learning."
                file_id = file_record.id
                session_id = 'default'
                context = ""
                
                result, status_code = call_claude_with_file(
                    user_id, message, file_id, session_id, context
                )
                # Update the description with Claude's response
                if status_code == 200:
                    # Extract just the response text, not the whole result dict
                    claude_response = result.get('response', '')
                    file_record.description = claude_response
                    db.session.commit()  # Save the updated description
                    
                    return jsonify({
                        'success': True, 
                        'file_id': file_record.id
                    }), status_code
                else:
                    # Handle error case
                    return jsonify({
                        'success': False,
                        'error': result.get('error', 'Unknown error')
                    }), status_code
            # file_record.description = str(result)
            
            # print("result file read: ", result)
            # return jsonify({'success': True, 'id': file_record.id})
            else:
                return jsonify({
                        'success': True, 
                        'file_id': file_record.id
                    }), 200
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/files/<int:file_id>/download')
# @requires_auth
def download_file(file_id):
    token = request.headers.get('Authorization')
    user_id = ""
    if token and token.startswith('Bearer '):
        token = token.split(' ')[1]
        user_id = verify_token_b(token)
    file = File.query.filter_by(id=file_id, user_id=user_id).first()
    if not file:
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(file.filepath, as_attachment=True, download_name=file.original_filename)

@app.route('/api/files/<int:file_id>', methods=['DELETE'])
# @requires_auth
def delete_file(file_id):
    token = request.headers.get('Authorization')
    if token and token.startswith('Bearer '):
        token = token.split(' ')[1]
        user_id = verify_token_b(token)
    file = File.query.filter_by(id=file_id, user_id=user_id).first()
    if not file:
        return jsonify({'error': 'File not found'}), 404
    
    # Delete physical file
    if os.path.exists(file.filepath):
        os.remove(file.filepath)
    
    db.session.delete(file)
    db.session.commit()
    
    return jsonify({'success': True})

@app.route('/api/links', methods=['GET', 'POST'])
# @requires_auth
def handle_links():
    token = request.headers.get('Authorization')
    if token and token.startswith('Bearer '):
        token = token.split(' ')[1]
        user_id = verify_token_b(token)
    with app.app_context():    
        user = User.query.get(user_id)
        if not user.claude_api_key:
            return jsonify({'error': 'Claude API key not configured'}), 400
        if request.method == 'GET':
            links = Link.query.filter_by(user_id=user_id).order_by(Link.created_at.desc()).all()
            return jsonify([{
                'id': link.id,
                'url': link.url,
                'title': link.title,
                'description': link.description,
                'favicon': link.favicon,
                'tags': json.loads(link.tags) if link.tags else [],
                'subject': link.subject,
                'created_at': link.created_at.isoformat()
            } for link in links])
        
        elif request.method == 'POST':
            data = request.get_json()
            url = data.get('url')
            
            if not url:
                return jsonify({'error': 'URL is required'}), 400
            
            # Extract metadata
            metadata = extract_metadata_from_url(url)
            
            # Example 2: Just summarize a URL
            summary = rag_system.summarize_url(
                api_key=user.claude_api_key,
                url=url,
                focus_query="Educational questions and pages only."
            )
            print("SUMMARY OF URL: ", summary)
            
            description_str = data.get('description', metadata['description'])
            joined_description = f"""{description_str}
                    Claude Summary: 
                    
                    {summary.get('response')}"""

            link = Link(
                url=url,
                title=data.get('title', metadata['title']),
                description=joined_description,
                favicon=metadata['favicon'],
                tags=json.dumps(data.get('tags', [])),
                subject=data.get('subject', ''),
                user_id=user_id
            )
            
            db.session.add(link)
            db.session.commit()
            
            return jsonify({'success': True, 'id': link.id})

@app.route('/api/links/<int:link_id>', methods=['DELETE'])
#@requires_auth
def delete_link(link_id):
    token = request.headers.get('Authorization')
    if token and token.startswith('Bearer '):
        token = token.split(' ')[1]
        user_id = verify_token_b(token)
    link = Link.query.filter_by(id=link_id, user_id=user_id).first()
    if not link:
        return jsonify({'error': 'Link not found'}), 404
    
    db.session.delete(link)
    db.session.commit()
    
    return jsonify({'success': True})

@app.route('/api/search', methods=['POST'])
#@requires_auth
def search():
    data = request.get_json()
    query = data.get('query', '')
    filters = data.get('filters', {})
    # Extract content types based on filters sent
    content_types = []
    for key in ['notes', 'files', 'links']:
        if filters.get(key):
            content_types.append(key)
    token = request.headers.get('Authorization')
    user_id = ""
    if token and token.startswith('Bearer '):
        token = token.split(' ')[1]
        user_id = verify_token_b(token)
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    user = User.query.get(user_id)
    if not user.claude_api_key:
        return jsonify({'error': 'Claude API key not configured'}), 400

    results = rag_system.search_content(user.claude_api_key, user_id, query, content_types)
    
    return jsonify([{
        'content': result.content,
        'source_type': result.source_type,
        'source_id': result.source_id,
        'title': result.title,
        'relevance_score': result.relevance_score,
        'created_at': result.created_at.isoformat()
    } for result in results])

@app.route('/api/claude-query', methods=['POST'])
#@requires_auth
def claude_query():
    data = request.get_json()
    query = data.get('query', '')
    token = request.headers.get('Authorization')
    user_id = ""
    if token and token.startswith('Bearer '):
        token = token.split(' ')[1]
        user_id = verify_token_b(token)
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    user = User.query.get(user_id)
    if not user.claude_api_key:
        return jsonify({'error': 'Claude API key not configured'}), 400
    
    # Search for relevant content
    results = rag_system.search_content(user.claude_api_key, user_id, query)
    
    # Query Claude with context
    response = rag_system.query_claude(user.claude_api_key, query, results)
    
    return jsonify({
        'response': response,
        'context_sources': [result.title
        for result in results]
    })

# Routes
@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data or not all(k in data for k in ('name', 'email', 'password')):
            return jsonify({'message': 'Missing required fields: name, email, password'}), 400
        
        name = data['name'].strip()
        email = data['email'].strip().lower()
        password = data['password']
        
        # Validate input
        if not name:
            return jsonify({'message': 'Name is required'}), 400
        
        if not validate_email(email):
            return jsonify({'message': 'Invalid email format'}), 400
        
        if not validate_password(password):
            return jsonify({'message': 'Password must be at least 6 characters long'}), 400
        
        # Check if user already exists
        with app.app_context():
            if User.query.filter_by(email=email).first():
                return jsonify({'message': 'Email already registered'}), 409
            
            # Create new user
            user = User(username=name, email=email)
            user.set_password(password)
            
            db.session.add(user)
            db.session.commit()
            
            return jsonify({
                'message': 'User created successfully',
                'user': user.to_dict()
            }), 201
        
    except Exception as e:
        db.session.rollback()
        print(e)
        return jsonify({'message': 'Registration failed. Please try again.'}), 500

# Helper functions
def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    """Password must be at least 6 characters long"""
    return len(password) >= 6

def generate_token(user_id):
    """Generate JWT token for user"""
    payload = {
        'user_id': user_id,
        'exp': datetime.now(timezone.utc) + timedelta(days=50),  # Token expires in 24 hours
        'iat': datetime.now(timezone.utc)
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

def verify_token(token):
    """Verify JWT token and return user_id"""
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return payload['user_id']
    except jwt.ExpiredSignatureError:
        print("Token Expired")
        return None
    except jwt.InvalidTokenError:
        print("Token Expired")
        return None
    
def verify_token_b(token):
    """Verify JWT token and return user_id"""
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'], options={"verify_signature": False, "verify_exp": False})
        return payload['user_id']
    except jwt.ExpiredSignatureError:
        print("Token Expired")
        return None
    except jwt.InvalidTokenError:
        print("Token Expired")
        return None

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data or not all(k in data for k in ('email', 'password')):
            return jsonify({'message': 'Missing email or password'}), 400
        
        email = data['email'].strip().lower()
        password = data['password']
        
        # Validate input
        if not validate_email(email):
            return jsonify({'message': 'Invalid email format'}), 400
        
        # Find user
        user = User.query.filter_by(email=email).first()
        
        if not user or not user.check_password(password):
            return jsonify({'message': 'Invalid email or password'}), 401
        
        # Generate token
        token = generate_token(user.id)
        
        return jsonify({
            'message': 'Login successful',
            'token': token,
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        print("Login Error: ", e)
        return jsonify({'message': 'Login failed. Please try again.'}), 500

@app.route('/api/settings', methods=['GET', 'POST'])
#@requires_auth
def handle_settings():
    token = request.headers.get('Authorization')
    user_id = ""
    if token and token.startswith('Bearer '):
        token = token.split(' ')[1]
        user_id = verify_token_b(token)
    user = User.query.get(user_id)
    
    if request.method == 'GET':
        return jsonify({
            'username': user.username,
            'email': user.email,
            'has_claude_api_key': bool(user.claude_api_key)
        })
    
    elif request.method == 'POST':
        data = request.get_json()
        
        if 'claude_api_key' in data:
            user.claude_api_key = data.get('claude_api_key')
        
        db.session.commit()
        return jsonify({'success': True})

@app.route('/api/dashboard-stats')
#@requires_auth
def dashboard_stats():
    token = request.headers.get('Authorization')
    if token and token.startswith('Bearer '):
        token = token.split(' ')[1]
        user_id = verify_token_b(token)
    # Get counts
    notes_count = Note.query.filter_by(user_id=user_id).count()
    files_count = File.query.filter_by(user_id=user_id).count()
    links_count = Link.query.filter_by(user_id=user_id).count()
    
    # Get todos
    todos = Note.query.filter_by(user_id=user_id, is_todo=True, is_completed=False).all()
    
    # Get recent activity
    recent_notes = Note.query.filter_by(user_id=user_id).order_by(Note.created_at.desc()).limit(5).all()
    recent_files = File.query.filter_by(user_id=user_id).order_by(File.created_at.desc()).limit(5).all()
    recent_links = Link.query.filter_by(user_id=user_id).order_by(Link.created_at.desc()).limit(5).all()
    
    return jsonify({
        'counts': {
            'notes': notes_count,
            'files': files_count,
            'links': links_count,
            'todos': len(todos)
        },
        'todos': [{
            'id': todo.id,
            'title': todo.title,
            'due_date': todo.due_date.isoformat() if todo.due_date else None,
            'priority': todo.priority
        } for todo in todos],
        'recent_activity': {
            'notes': [{'id': n.id, 'title': n.title, 'created_at': n.created_at.isoformat()} for n in recent_notes],
            'files': [{'id': f.id, 'title': f.original_filename, 'created_at': f.created_at.isoformat()} for f in recent_files],
            'links': [{'id': l.id, 'title': l.title, 'created_at': l.created_at.isoformat()} for l in recent_links]
        }
    })

# Initialize database

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)