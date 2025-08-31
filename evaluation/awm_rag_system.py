import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
import pickle
from pathlib import Path
import hashlib

# Check for optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import openai
    from openai import OpenAI
    client = OpenAI()
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class ProductionRAGSystem:
    """Production-ready RAG system with robust error handling"""
    
    def __init__(self, memory_file: str = "awm_memories.json", 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 use_openai_embeddings: bool = False):
        self.memory_file = memory_file
        self.embedding_cache_file = memory_file.replace('.json', '_embeddings.pkl')
        self.workflows_file = memory_file.replace('.json', '_workflows.txt')
        self.use_openai_embeddings = use_openai_embeddings
        
        # Initialize embedding system with fallbacks
        self.embedding_model = None
        self.embedding_dim = 384  # Default dimension
        self._init_embedding_system(embedding_model)
        
        # Load cached data
        self.workflow_embeddings = self._load_embeddings_cache()
        self.workflows_data = []
        self.last_workflows_mtime = 0
        self._load_workflows()
    
    def _init_embedding_system(self, embedding_model: str):
        """Initialize embedding system with proper fallbacks"""
        if self.use_openai_embeddings and OPENAI_AVAILABLE:
            try:
                # Test OpenAI connectivity
                test_response = client.embeddings.create(
                    input="test",
                    model="ttext-embedding-3-small"
                )
                self.embedding_model = None
                self.embedding_dim = 1536
                print("Using OpenAI embeddings")
                return
            except Exception as e:
                print(f"OpenAI embeddings failed, falling back to local: {e}")
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                print(f"Using local embeddings: {embedding_model}")
                return
            except Exception as e:
                print(f"Local embeddings failed: {e}")
        
        # Ultimate fallback: keyword-based similarity
        print("Warning: Using keyword-based similarity fallback")
        self.embedding_model = "keyword_fallback"
        self.embedding_dim = 0
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with robust error handling"""
        if self.embedding_model == "keyword_fallback":
            # Simple keyword-based vector (fallback)
            keywords = text.lower().split()
            # Create simple keyword hash vector
            hash_vec = [hash(word) % 1000 for word in keywords[:10]]
            while len(hash_vec) < 10:
                hash_vec.append(0)
            return np.array(hash_vec[:10], dtype=float)
        
        if self.use_openai_embeddings and OPENAI_AVAILABLE:
            try:
                response = client.embeddings.create(
                    input=text[:8000],  # Limit text length
                    model="text-embedding-3-small"
                )
                return np.array(response.data[0].embedding)
            except Exception as e:
                print(f"OpenAI embedding failed: {e}")
                # Fall back to local if available
                if self.embedding_model and self.embedding_model != "keyword_fallback":
                    return self.embedding_model.encode(text)
                raise
        else:
            if self.embedding_model and self.embedding_model != "keyword_fallback":
                return self.embedding_model.encode(text)
            else:
                return self._get_keyword_fallback(text)
    
    def _load_embeddings_cache(self) -> Dict[str, np.ndarray]:
        """Load cached embeddings with error handling"""
        if not os.path.exists(self.embedding_cache_file):
            return {}
            
        try:
            with open(self.embedding_cache_file, 'rb') as f:
                cache = pickle.load(f)
                print(f"Loaded {len(cache)} cached embeddings")
                return cache
        except Exception as e:
            print(f"Error loading embeddings cache, starting fresh: {e}")
            return {}
    
    def _save_embeddings_cache(self) -> bool:
        """Save embeddings to cache with error handling"""
        try:
            with open(self.embedding_cache_file, 'wb') as f:
                pickle.dump(self.workflow_embeddings, f)
            return True
        except Exception as e:
            print(f"Warning: Could not save embeddings cache: {e}")
            return False
    
    def _parse_workflows(self, workflows_text: str) -> List[Dict[str, str]]:
        """Parse workflows text with robust error handling"""
        workflows = []
        
        try:
            # Split by double newlines to get sections
            sections = [s.strip() for s in workflows_text.split('\n\n') if s.strip()]
            
            current_workflow = None
            for section in sections:
                lines = section.split('\n')
                first_line = lines[0].strip()
                
                # Detect workflow headers
                if first_line.startswith('##') and not first_line.startswith('###'):
                    # Save previous workflow
                    if current_workflow and current_workflow.get('name'):
                        workflows.append(current_workflow)
                    
                    # Start new workflow
                    workflow_name = first_line.replace('##', '').strip()
                    current_workflow = {
                        'name': workflow_name,
                        'description': '',
                        'steps': [],
                        'full_text': section,
                        'searchable_text': section.lower()
                    }
                    
                    # Extract description and steps
                    for line in lines[1:]:
                        line = line.strip()
                        if not line:
                            continue
                            
                        # First non-action line is likely description
                        if (not current_workflow['description'] and 
                            not any(line.startswith(prefix) for prefix in 
                                   ['[', 'CLICK', 'TYPE', 'SELECT', 'Click', 'Fill', 'Navigate', 'Run', 'Read', 'Edit', 'Browse'])):
                            current_workflow['description'] = line
                        
                        # Collect action steps
                        if any(line.startswith(prefix) for prefix in 
                              ['[', 'CLICK', 'TYPE', 'SELECT', 'Click', 'Fill', 'Navigate', 'Run', 'Read', 'Edit', 'Browse']):
                            current_workflow['steps'].append(line)
                
                elif current_workflow:
                    # Continue current workflow
                    current_workflow['full_text'] += '\n\n' + section
                    current_workflow['searchable_text'] += ' ' + section.lower()
                    
                    for line in lines:
                        line = line.strip()
                        if any(line.startswith(prefix) for prefix in 
                              ['[', 'CLICK', 'TYPE', 'SELECT', 'Click', 'Fill', 'Navigate', 'Run', 'Read', 'Edit', 'Browse']):
                            current_workflow['steps'].append(line)
            
            # Add final workflow
            if current_workflow and current_workflow.get('name'):
                workflows.append(current_workflow)
                
        except Exception as e:
            print(f"Error parsing workflows: {e}")
            return []
        
        return workflows
    
    def _should_reload_workflows(self) -> bool:
        """Check if workflows file has been updated"""
        if not os.path.exists(self.workflows_file):
            return False
            
        try:
            current_mtime = os.path.getmtime(self.workflows_file)
            return current_mtime > self.last_workflows_mtime
        except OSError:
            return False
    
    def _load_workflows(self):
        """Load and parse workflows with automatic reloading"""
        if not os.path.exists(self.workflows_file):
            return
        
        # Skip if file hasn't changed
        if not self._should_reload_workflows() and self.workflows_data:
            return
        
        try:
            with open(self.workflows_file, 'r') as f:
                workflows_text = f.read()
            
            self.workflows_data = self._parse_workflows(workflows_text)
            self.last_workflows_mtime = os.path.getmtime(self.workflows_file)
            
            print(f"Loaded {len(self.workflows_data)} workflows")
            
            # Generate embeddings for new workflows
            if self.workflows_data:
                self._update_embeddings()
            
        except Exception as e:
            print(f"Error loading workflows: {e}")
            self.workflows_data = []
    
    def _create_searchable_text(self, workflow: Dict[str, str]) -> str:
        """Create searchable text from workflow data"""
        parts = [
            workflow.get('name', ''),
            workflow.get('description', ''),
            workflow.get('searchable_text', ''),
            ' '.join(workflow.get('steps', []))
        ]
        return ' '.join(filter(None, parts))
    
    def _update_embeddings(self):
        """Update embeddings for new workflows"""
        if self.embedding_model == "keyword_fallback":
            return  # Skip embedding generation for fallback
            
        new_embeddings_count = 0
        
        for i, workflow in enumerate(self.workflows_data):
            # Create stable workflow ID
            workflow_content = workflow.get('name', '') + workflow.get('description', '')
            workflow_id = f"workflow_{hashlib.md5(workflow_content.encode()).hexdigest()[:8]}"
            
            if workflow_id not in self.workflow_embeddings:
                try:
                    searchable_text = self._create_searchable_text(workflow)
                    embedding = self._get_embedding(searchable_text)
                    self.workflow_embeddings[workflow_id] = embedding
                    workflow['embedding_id'] = workflow_id
                    new_embeddings_count += 1
                except Exception as e:
                    print(f"Error generating embedding for {workflow.get('name', 'Unknown')}: {e}")
                    continue
        
        if new_embeddings_count > 0:
            self._save_embeddings_cache()
            print(f"Generated {new_embeddings_count} new embeddings")
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity with error handling"""
        try:
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return np.dot(a, b) / (norm_a * norm_b)
        except Exception:
            return 0.0
    
    def _keyword_similarity(self, query: str, workflow_text: str) -> float:
        """Fallback keyword-based similarity"""
        query_words = set(query.lower().split())
        workflow_words = set(workflow_text.lower().split())
        
        if not query_words or not workflow_words:
            return 0.0
            
        intersection = query_words.intersection(workflow_words)
        union = query_words.union(workflow_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def retrieve_relevant_workflows(self, query: str, top_k: int = 3, 
                                  similarity_threshold: float = 0.25) -> List[Dict[str, any]]:
        """Retrieve relevant workflows with robust error handling"""
        # Refresh workflows if needed
        self._load_workflows()
        
        if not self.workflows_data:
            return []
        
        similarities = []
        
        try:
            if self.embedding_model == "keyword_fallback":
                # Use keyword-based similarity
                for workflow in self.workflows_data:
                    searchable_text = self._create_searchable_text(workflow)
                    similarity = self._keyword_similarity(query, searchable_text)
                    similarities.append((workflow, similarity))
            else:
                # Use embedding-based similarity
                query_embedding = self._get_embedding(query)
                
                for workflow in self.workflows_data:
                    embedding_id = workflow.get('embedding_id')
                    if embedding_id and embedding_id in self.workflow_embeddings:
                        workflow_embedding = self.workflow_embeddings[embedding_id]
                        similarity = self._cosine_similarity(query_embedding, workflow_embedding)
                        similarities.append((workflow, similarity))
        
        except Exception as e:
            print(f"Error in similarity calculation: {e}")
            return []
        
        # Sort and filter results
        similarities.sort(key=lambda x: x[1], reverse=True)
        relevant_workflows = []
        
        for workflow, similarity in similarities[:top_k]:
            if similarity >= similarity_threshold:
                relevant_workflows.append({
                    'workflow': workflow,
                    'similarity': similarity,
                    'name': workflow.get('name', 'Unknown'),
                    'description': workflow.get('description', ''),
                    'full_text': workflow.get('full_text', '')
                })
        
        return relevant_workflows
    
    def format_workflows_for_prompt(self, relevant_workflows: List[Dict[str, any]]) -> str:
        """Format retrieved workflows for agent prompt"""
        if not relevant_workflows:
            return ""
        
        formatted = "## Relevant Learned Workflows\n\n"
        formatted += "The following workflows were learned from previous successful task executions. "
        formatted += "Use them as patterns when applicable:\n\n"
        
        for item in relevant_workflows:
            workflow = item['workflow']
            similarity = item['similarity']
            
            formatted += f"### {workflow.get('name', 'Workflow')} (relevance: {similarity:.2f})\n"
            if workflow.get('description'):
                formatted += f"{workflow['description']}\n\n"
            
            # Include workflow steps or full text
            if workflow.get('full_text'):
                formatted += workflow['full_text'] + "\n\n"
            elif workflow.get('steps'):
                formatted += "\n".join(workflow['steps']) + "\n\n"
        
        return formatted.strip()
    
    def get_workflow_suggestions(self, task_instruction: str, top_k: int = 3) -> str:
        """Main function to get workflow suggestions"""
        try:
            relevant = self.retrieve_relevant_workflows(task_instruction, top_k=top_k)
            return self.format_workflows_for_prompt(relevant)
        except Exception as e:
            print(f"Error getting workflow suggestions: {e}")
            return ""
    
    def get_system_status(self) -> Dict[str, any]:
        """Get system status for monitoring"""
        status = {
            "embedding_system": "unknown",
            "workflows_loaded": len(self.workflows_data),
            "cached_embeddings": len(self.workflow_embeddings),
            "workflows_file_exists": os.path.exists(self.workflows_file),
            "last_error": None
        }
        
        if self.embedding_model == "keyword_fallback":
            status["embedding_system"] = "keyword_fallback"
        elif self.use_openai_embeddings:
            status["embedding_system"] = "openai"
        elif self.embedding_model:
            status["embedding_system"] = "local_transformers"
            
        return status


# Global RAG instance with error handling
try:
    rag_system = ProductionRAGSystem()
except Exception as e:
    print(f"Warning: Could not initialize RAG system: {e}")
    rag_system = None


def get_relevant_workflows_rag(task_instruction: str, top_k: int = 3) -> str:
    """Main function to get relevant workflows using RAG with fallback"""
    if rag_system is None:
        print("RAG system not available")
        return ""
        
    try:
        return rag_system.get_workflow_suggestions(task_instruction, top_k=top_k)
    except Exception as e:
        print(f"Error in RAG retrieval: {e}")
        return ""