import os
import json
import hashlib
import fcntl
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

import openai
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

class AWMMemorySystem:
    """Production-ready Agent Workflow Memory system for TheAgentCompany"""
    
    def __init__(self, memory_file: str = None, model: str = "gpt-4o"):
        if memory_file is None:
            # Use evaluation directory as default
            eval_dir = os.path.dirname(os.path.abspath(__file__))
            memory_file = os.path.join(eval_dir, "awm_memories.json")
            
        self.memory_file = memory_file
        self.model = model
        self.memories = self._load_memories()
        
        # AWM prompts adapted for TheAgentCompany
        self.instruction = """Given a list of professional work tasks, extract common workflows to solve these tasks.
Each task contains a natural language instruction and a series of actions. Find repetitive action patterns across tasks and extract them as reusable workflows.
Each workflow should be a commonly-reused sub-routine. Avoid similar or overlapping workflows. Each workflow needs at least two steps."""

        self.one_shot_example = """Website: Professional Work Environment (GitLab, OwnCloud, RocketChat, Plane)
## Query 1: Create a Python file and test it
Actions:
1. Read task instruction file
2. Create new Python file with code
3. Make file executable with chmod
4. Test file by running it
5. Verify output is correct

## Query 2: Upload document to OwnCloud
Actions:
1. Browse to OwnCloud URL
2. Login with credentials
3. Navigate to target folder
4. Upload file from local system
5. Verify upload completion

Summary Workflows:
## create_and_test_file
Given that you need to create and test a script file:
Read the task requirements from instruction file
Create new file with appropriate content: {file-path}
Make file executable: chmod +x {file-path}
Test the file by executing it
Verify the results are correct

## upload_file_to_service
Given that you need to upload a file to a web service:
Browse to the service URL: {service-url}
Login with provided credentials
Navigate to destination folder
Upload the file: {file-path}
Verify upload was successful"""

    def _load_memories(self) -> List[Dict[str, Any]]:
        """Load existing memories from file with error handling"""
        if not os.path.exists(self.memory_file):
            return []
            
        try:
            with open(self.memory_file, 'r') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
                data = json.load(f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, IOError, OSError) as e:
            print(f"Error loading memories: {e}")
            return []

    def _save_memories(self) -> bool:
        """Save memories to file with locking and error handling"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            
            with open(self.memory_file, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock for writing
                json.dump(self.memories, f, indent=2)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return True
        except Exception as e:
            print(f"Error saving memories: {e}")
            return False

    def _generate_memory_hash(self, task_instruction: str, actions: List[str]) -> str:
        """Generate a unique hash for the memory to avoid duplicates"""
        content = task_instruction + "".join(actions)
        return hashlib.md5(content.encode()).hexdigest()

    def _extract_actions_from_state(self, state_obj) -> List[str]:
        """Extract meaningful actions from OpenHands State object"""
        actions = []
        
        # Handle direct State object or its history
        if hasattr(state_obj, 'history'):
            history = state_obj.history
        elif isinstance(state_obj, list):
            history = state_obj
        else:
            return actions
            
        for event in history:
            try:
                # Skip initial messages and recalls
                if hasattr(event, 'action'):
                    action_type = event.action
                    
                    # Extract different action types
                    if action_type == 'read':
                        if hasattr(event, 'path'):
                            actions.append(f"Read file: {event.path}")
                            
                    elif action_type == 'edit':
                        if hasattr(event, 'path'):
                            actions.append(f"Edit file: {event.path}")
                            
                    elif action_type == 'run':
                        if hasattr(event, 'command'):
                            # Filter out noise commands
                            cmd = str(event.command).strip()
                            if cmd and not cmd.startswith('cat -n') and len(cmd) < 200:
                                actions.append(f"Run command: {cmd}")
                                
                    elif action_type == 'browse':
                        if hasattr(event, 'url'):
                            actions.append(f"Browse to: {event.url}")
                            
                    elif action_type == 'message':
                        if hasattr(event, 'content'):
                            content = str(event.content)
                            # Skip system messages
                            if not content.startswith('Please continue') and len(content) < 100:
                                actions.append(f"Message: {content}")
                                
                    elif action_type == 'finish':
                        if hasattr(event, 'task_completed') and event.task_completed == 'true':
                            actions.append("Task completed successfully")
                            
                # Handle string representations of events
                elif isinstance(event, str):
                    if 'Action' in event and len(event) < 500:
                        actions.append(f"Action: {event}")
                        
            except Exception as e:
                # Skip malformed events
                continue
                
        return actions

    def _format_trajectory_for_induction(self, task_instruction: str, actions: List[str], 
                                       task_name: str = None) -> str:
        """Format trajectory data for workflow induction"""
        header = f"## Query: {task_instruction}\nActions:\n"
        action_lines = [f"{i}. {action}" for i, action in enumerate(actions, 1)]
        
        formatted = header + "\n".join(action_lines)
        
        if task_name:
            formatted = f"Task: {task_name}\n" + formatted
            
        return formatted

    def _detect_task_success(self, state_obj) -> bool:
        """Detect if task completed successfully"""
        if hasattr(state_obj, 'agent_state'):
            # Check if agent finished successfully
            if str(state_obj.agent_state) == 'AgentState.FINISHED':
                return True
                
        if hasattr(state_obj, 'history'):
            # Look for successful completion indicators
            for event in reversed(state_obj.history):  # Check recent events first
                if hasattr(event, 'action') and event.action == 'finish':
                    if hasattr(event, 'task_completed') and event.task_completed == 'true':
                        return True
                        
        return False

    def add_trajectory_to_memory(self, task_instruction: str, state_obj, 
                                task_name: str = None, dependencies: List[str] = None,
                                force_success: bool = False):
        """Convert completed trajectory to memory and store it"""
        
        # Check task success unless forced
        if not force_success:
            task_success = self._detect_task_success(state_obj)
            if not task_success:
                print(f"Skipping failed task: {task_name}")
                return False
        
        # Extract actions from state
        actions = self._extract_actions_from_state(state_obj)
        
        if len(actions) < 2:  # Need at least 2 actions for meaningful workflow
            print(f"Insufficient actions ({len(actions)}) for task: {task_name}")
            return False
        
        # Generate unique hash
        memory_hash = self._generate_memory_hash(task_instruction, actions)
        
        # Check if memory already exists
        if any(mem.get("hash") == memory_hash for mem in self.memories):
            print(f"Memory already exists for task: {task_name}")
            return False
        
        # Create memory entry
        memory_entry = {
            "hash": memory_hash,
            "task_name": task_name,
            "task_instruction": task_instruction,
            "actions": actions,
            "dependencies": dependencies or [],
            "trajectory_formatted": self._format_trajectory_for_induction(
                task_instruction, actions, task_name
            ),
            "timestamp": int(time.time()),
            "success": True,
            "action_count": len(actions)
        }
        
        # Add to memories
        self.memories.append(memory_entry)
        
        # Save to file
        if self._save_memories():
            print(f"Added trajectory to AWM memory: {task_name} ({len(actions)} actions)")
            return True
        else:
            # Remove from memory if save failed
            self.memories.pop()
            return False

    def _validate_workflows(self, workflows_text: str) -> bool:
        """Validate that generated workflows are well-formed"""
        if not workflows_text or len(workflows_text.strip()) < 100:
            return False
            
        # Check for workflow structure markers
        if '##' not in workflows_text or 'Given that' not in workflows_text:
            return False
            
        # Check for reasonable length (not too short or extremely long)
        if len(workflows_text) > 10000:
            return False
            
        return True

    def induce_workflows_from_memories(self, min_memories: int = 3, max_examples: int = 8) -> str:
        """Induce workflows from stored memories using LLM with validation"""
        
        if len(self.memories) < min_memories:
            print(f"Not enough memories ({len(self.memories)}) for workflow induction. Need at least {min_memories}.")
            return ""
        
        # Select recent successful memories for induction
        successful_memories = [m for m in self.memories if m.get("success", True)]
        if len(successful_memories) < min_memories:
            print(f"Not enough successful memories ({len(successful_memories)}) for induction.")
            return ""
            
        # Use recent memories but ensure diversity
        recent_memories = successful_memories[-max_examples:]
        
        # Format memories for prompting
        formatted_examples = [mem["trajectory_formatted"] for mem in recent_memories]
        examples_text = "\n\n".join(formatted_examples)
        
        # Create prompt for workflow induction
        prompt = f"{self.instruction}\n\n{self.one_shot_example}\n\n{examples_text}\n\nSummary Workflows:"
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2048,
                timeout=30  # Add timeout
            )
            
            workflows = response.choices[0].message.content
            
            # Validate workflows before saving
            if not self._validate_workflows(workflows):
                print("Generated workflows failed validation")
                return ""
            
            # Save induced workflows
            workflows_file = self.memory_file.replace('.json', '_workflows.txt')
            try:
                with open(workflows_file, 'w') as f:
                    f.write(workflows)
                
                print(f"Induced workflows from {len(recent_memories)} memories. Saved to: {workflows_file}")
                return workflows
            except Exception as e:
                print(f"Error saving workflows: {e}")
                return ""
                
        except Exception as e:
            print(f"Error inducing workflows: {e}")
            return ""

    def get_relevant_workflows(self, task_instruction: str, use_rag: bool = True, top_k: int = 3) -> str:
        """Get relevant workflows with graceful fallback"""
        workflows_file = self.memory_file.replace('.json', '_workflows.txt')
        
        if not os.path.exists(workflows_file):
            return ""
        
        if use_rag:
            try:
                from awm_rag_system import get_relevant_workflows_rag
                result = get_relevant_workflows_rag(task_instruction, top_k=top_k)
                if result:
                    return result
            except Exception as e:
                print(f"RAG retrieval failed, using fallback: {e}")
        
        # Fallback: return all workflows with basic formatting
        try:
            with open(workflows_file, 'r') as f:
                content = f.read().strip()
                if content:
                    return f"## Available Learned Workflows\n\nThe following workflows have been learned from previous successful task executions:\n\n{content}"
        except Exception as e:
            print(f"Error reading workflows file: {e}")
            
        return ""

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories"""
        if not self.memories:
            return {"total_memories": 0, "status": "empty"}
        
        # Calculate statistics
        total_actions = sum(mem.get("action_count", 0) for mem in self.memories)
        dependencies = [dep for mem in self.memories for dep in mem.get("dependencies", [])]
        dependency_counts = {}
        for dep in dependencies:
            dependency_counts[dep] = dependency_counts.get(dep, 0) + 1
        
        recent_tasks = [mem.get("task_name", "Unknown") for mem in self.memories[-5:]]
        avg_actions = total_actions / len(self.memories) if self.memories else 0
        
        return {
            "total_memories": len(self.memories),
            "total_actions": total_actions,
            "average_actions_per_memory": round(avg_actions, 2),
            "dependency_distribution": dependency_counts,
            "recent_task_names": recent_tasks,
            "memory_file_size": os.path.getsize(self.memory_file) if os.path.exists(self.memory_file) else 0,
            "status": "healthy"
        }

    def cleanup_old_memories(self, max_memories: int = 100):
        """Remove oldest memories to prevent unbounded growth"""
        if len(self.memories) > max_memories:
            removed_count = len(self.memories) - max_memories
            self.memories = self.memories[-max_memories:]  # Keep most recent
            if self._save_memories():
                print(f"Cleaned up {removed_count} old memories, kept {max_memories} most recent")
                return True
        return False


# Global instance for easy access
awm_memory = AWMMemorySystem()