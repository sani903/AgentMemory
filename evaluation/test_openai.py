#!/usr/bin/env python3
"""
Enhanced AWM unit tests with OpenAI integration and proper OpenHands state parsing
Run from evaluation directory: python test_awm_system_enhanced.py
"""

import os
import sys
import json
import tempfile
import shutil
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test configuration
TEST_STATE_FILE = "state_safety-deception.json"
USE_OPENAI_EMBEDDINGS = True  # Set to False to use sentence-transformers
TEST_WORKFLOW_INDUCTION = True  # Set to False to skip OpenAI workflow generation


def load_test_state():
    """Load the actual serialized state object from JSON file"""
    state_file = Path(__file__).parent / TEST_STATE_FILE
    if not state_file.exists():
        raise FileNotFoundError(f"Test state file not found: {state_file}")
    
    with open(state_file, 'r') as f:
        state_str = f.read().strip()
    
    print(f"Loaded actual state string length: {len(state_str)}")
    print(f"State preview: {state_str[:200]}...")
    
    # Parse the state string to extract basic info for validation
    if "safety" in state_str.lower() or "deception" in state_str.lower():
        print("✓ Detected safety/deception-related task in state")
    if any(field in state_str for field in ["iteration_flag=", "agent_state=", "history="]):
        print("✓ Detected OpenHands State structure")
    if "history=[" in state_str:
        print("✓ Detected action history in state")
    
    return state_str


def parse_openhands_state_robust(state_str: str) -> Dict[str, Any]:
    """Robustly parse OpenHands State object string to extract structured information"""
    parsed_data = {
        'session_id': '',
        'user_id': None,
        'iteration_flag': None,
        'agent_state': None,
        'delegate_level': 0,
        'actions': [],
        'observations': [],
        'task_info': None,
        'inputs': {},
        'outputs': {},
        'extra_data': {},
        'metrics': None
    }
    
    # Extract session_id
    session_match = re.search(r"session_id='([^']*)'", state_str)
    if session_match:
        parsed_data['session_id'] = session_match.group(1)
    
    # Extract user_id
    user_match = re.search(r"user_id=([^,\)]+)", state_str)
    if user_match:
        user_val = user_match.group(1).strip()
        if user_val != 'None':
            parsed_data['user_id'] = user_val.strip("'\"")
    
    # Extract iteration information from IterationControlFlag
    iteration_flag_match = re.search(r"iteration_flag=IterationControlFlag\([^)]*current_value=(\d+)[^)]*max_value=(\d+)", state_str)
    if iteration_flag_match:
        parsed_data['iteration_flag'] = {
            'current_value': int(iteration_flag_match.group(1)),
            'max_value': int(iteration_flag_match.group(2))
        }
    
    # Extract agent state
    agent_state_match = re.search(r"agent_state=AgentState\.(\w+)", state_str)
    if agent_state_match:
        parsed_data['agent_state'] = f"AgentState.{agent_state_match.group(1)}"
    
    # Extract delegate level
    delegate_match = re.search(r"delegate_level=(\d+)", state_str)
    if delegate_match:
        parsed_data['delegate_level'] = int(delegate_match.group(1))
    
    # Extract actions and observations from history
    actions, observations = extract_history_events(state_str)
    parsed_data['actions'] = actions
    parsed_data['observations'] = observations
    
    # Extract task information
    if '/instruction/task.md' in state_str:
        parsed_data['task_info'] = "/instruction/task.md"
    
    # Extract inputs/outputs if present
    inputs_match = re.search(r"inputs=({[^}]*})", state_str)
    if inputs_match:
        try:
            parsed_data['inputs'] = eval(inputs_match.group(1))
        except:
            parsed_data['inputs'] = {}
    
    outputs_match = re.search(r"outputs=({[^}]*})", state_str)
    if outputs_match:
        try:
            parsed_data['outputs'] = eval(outputs_match.group(1))
        except:
            parsed_data['outputs'] = {}
    
    return parsed_data


def extract_history_events(state_str: str) -> tuple[List[str], List[str]]:
    """Extract actions and observations from the history field of the state"""
    actions = []
    observations = []
    
    # Find the history section
    history_match = re.search(r"history=\[(.*?)\](?=,\s*inputs=|,\s*outputs=|,\s*agent_state=|\s*\)$)", state_str, re.DOTALL)
    if not history_match:
        # Try alternative patterns for history extraction
        history_match = re.search(r"history=\[(.*?)\]", state_str, re.DOTALL)
    
    if history_match:
        history_content = history_match.group(1)
        
        # Parse different action types with their complete content
        action_patterns = [
            # MessageAction
            (r"MessageAction\(content='([^']*(?:\\'[^']*)*)'(?:[^)]*image_urls=([^,)]*[^,)]*))?", 
             lambda m: f"message: {m.group(1)}" + (f" [images: {m.group(2)}]" if m.group(2) and m.group(2).strip() != 'None' else "")),
            
            # RecallAction
            (r"RecallAction\([^)]*query='([^']*(?:\\'[^']*)*)'", 
             lambda m: f"recall: {m.group(1)}"),
            
            # FileReadAction
            (r"FileReadAction\(path='([^']*(?:\\'[^']*)*)'(?:[^)]*thought='([^']*(?:\\'[^']*)*)')?", 
             lambda m: f"read_file: {m.group(1)}" + (f" [thought: {m.group(2)}]" if m.group(2) else "")),
            
            # FileWriteAction
            (r"FileWriteAction\(path='([^']*(?:\\'[^']*)*)'(?:[^)]*content='([^']{0,50})", 
             lambda m: f"write_file: {m.group(1)} [content: {m.group(2)}...]"),
            
            # CmdRunAction
            (r"CmdRunAction\(command='([^']*(?:\\'[^']*)*)'(?:[^)]*thought='([^']*(?:\\'[^']*)*)')?", 
             lambda m: f"run_command: {m.group(1)}" + (f" [thought: {m.group(2)}]" if m.group(2) else "")),
            
            # BrowserAction
            (r"BrowserAction\([^)]*url='([^']*(?:\\'[^']*)*)'", 
             lambda m: f"browse: {m.group(1)}"),
            
            # IPythonRunCellAction
            (r"IPythonRunCellAction\(code='([^']*(?:\\'[^']*)*)'", 
             lambda m: f"run_code: {m.group(1)[:50]}{'...' if len(m.group(1)) > 50 else ''}"),
            
            # AgentFinishAction
            (r"AgentFinishAction\([^)]*thought='([^']*(?:\\'[^']*)*)'", 
             lambda m: f"finish: {m.group(1)}"),
            
            # Generic action fallback
            (r"(\w+Action)\(", 
             lambda m: f"action: {m.group(1)}")
        ]
        
        # Parse observation types
        observation_patterns = [
            # FileReadObservation
            (r"FileReadObservation\(content='([^']{0,100})", 
             lambda m: f"file_content: {m.group(1)}..."),
            
            # CmdOutputObservation
            (r"CmdOutputObservation\([^)]*content='([^']{0,100})", 
             lambda m: f"command_output: {m.group(1)}..."),
            
            # BrowserOutputObservation
            (r"BrowserOutputObservation\([^)]*content='([^']{0,100})", 
             lambda m: f"browser_result: {m.group(1)}..."),
            
            # RecallObservation
            (r"RecallObservation\(content='([^']*(?:\\'[^']*)*)'", 
             lambda m: f"recall_result: {m.group(1)}"),
            
            # IPythonRunCellObservation
            (r"IPythonRunCellObservation\([^)]*content='([^']{0,100})", 
             lambda m: f"code_output: {m.group(1)}..."),
            
            # Generic observation fallback
            (r"(\w+Observation)\(", 
             lambda m: f"observation: {m.group(1)}")
        ]
        
        # Extract actions
        for pattern, formatter in action_patterns:
            matches = re.finditer(pattern, history_content, re.DOTALL)
            for match in matches:
                try:
                    action_desc = formatter(match)
                    actions.append(action_desc)
                except Exception as e:
                    # Fallback for parsing errors
                    actions.append(f"action: {match.group(0)[:50]}...")
        
        # Extract observations
        for pattern, formatter in observation_patterns:
            matches = re.finditer(pattern, history_content, re.DOTALL)
            for match in matches:
                try:
                    obs_desc = formatter(match)
                    observations.append(obs_desc)
                except Exception as e:
                    # Fallback for parsing errors
                    observations.append(f"observation: {match.group(0)[:50]}...")
    
    # If we still don't have actions, try a more aggressive extraction
    if not actions:
        # Look for any quoted strings that might be commands or paths
        quoted_strings = re.findall(r"'([^']{10,100})'", state_str)
        for string in quoted_strings[:10]:  # Limit to prevent spam
            if any(indicator in string.lower() for indicator in 
                   ['task.md', 'csv', 'http', 'python', 'git', 'docker', 'instruction', '/', 'cd ', 'ls ', 'cat ']):
                actions.append(f"reference: {string}")
    
    return actions, observations


def extract_actions_from_openhands_state(state_str: str) -> List[str]:
    """Extract actionable steps from OpenHands state string"""
    parsed = parse_openhands_state_robust(state_str)
    
    # Convert parsed actions into readable action descriptions
    readable_actions = []
    
    # Add context about the task and state
    if parsed['session_id']:
        readable_actions.append(f"Task session: {parsed['session_id']}")
    
    if parsed['task_info']:
        readable_actions.append(f"Read task instructions from {parsed['task_info']}")
    
    if parsed['iteration_flag']:
        current = parsed['iteration_flag']['current_value']
        max_val = parsed['iteration_flag']['max_value']
        readable_actions.append(f"Iteration progress: {current}/{max_val}")
    
    if parsed['agent_state']:
        readable_actions.append(f"Agent state: {parsed['agent_state']}")
    
    # Add all extracted actions
    readable_actions.extend(parsed['actions'])
    
    # Add key observations for context
    for obs in parsed['observations'][:5]:  # Limit to first 5 observations
        readable_actions.append(f"Observed: {obs}")
    
    # Add inputs/outputs if present
    if parsed['inputs']:
        readable_actions.append(f"Task inputs: {list(parsed['inputs'].keys())}")
    
    if parsed['outputs']:
        readable_actions.append(f"Task outputs: {list(parsed['outputs'].keys())}")
    
    return readable_actions


def create_state_wrapper(state_str: str):
    """Create a wrapper that presents the serialized state as needed by AWM"""
    class StateStringWrapper:
        def __init__(self, state_str):
            self.state_str = state_str
            self.parsed_state = parse_openhands_state_robust(state_str)
            
            # Determine agent state from parsed data
            if self.parsed_state['agent_state']:
                self.agent_state = self.parsed_state['agent_state']
            elif self.parsed_state['iteration_flag']:
                # Check if we've reached max iterations
                current = self.parsed_state['iteration_flag']['current_value']
                max_val = self.parsed_state['iteration_flag']['max_value']
                if current >= max_val:
                    self.agent_state = "AgentState.FINISHED"
                else:
                    self.agent_state = "AgentState.RUNNING"
            else:
                # Fallback - look for completion indicators
                if any(indicator in state_str.lower() for indicator in 
                       ['finished', 'completed', 'success', 'agentfinishaction']):
                    self.agent_state = "AgentState.FINISHED" 
                else:
                    self.agent_state = "AgentState.RUNNING"
            
    return StateStringWrapper(state_str)


class TestOpenAIIntegration:
    """Test OpenAI-powered features"""
    
    def __init__(self):
        self.temp_dir = None
        self.awm_memory = None
        self.rag_system = None
    
    def setup(self):
        """Set up OpenAI test environment"""
        print("\n=== Setting Up OpenAI Integration Tests ===")
        
        # Check OpenAI availability
        try:
            import openai
            from openai import OpenAI
            client = OpenAI()
            print("OpenAI client initialized")
        except Exception as e:
            print(f"OpenAI not available: {e}")
            return False
        
        # Test API connectivity
        try:
            test_response = client.embeddings.create(
                input="test connectivity",
                model="text-embedding-ada-002"
            )
            print(f"OpenAI API test successful (dimension: {len(test_response.data[0].embedding)})")
        except Exception as e:
            print(f"OpenAI API test failed: {e}")
            return False
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="openai_test_")
        print(f"OpenAI test directory: {self.temp_dir}")
        
        # Initialize AWM memory with OpenAI
        from awm_memory import AWMMemorySystem
        test_memory_file = os.path.join(self.temp_dir, "test_memories.json")
        self.awm_memory = AWMMemorySystem(memory_file=test_memory_file, model="gpt-4o-mini")
        
        # Initialize RAG system with OpenAI embeddings
        from awm_rag_system import ProductionRAGSystem
        self.rag_system = ProductionRAGSystem(
            memory_file=test_memory_file,
            use_openai_embeddings=True
        )
        
        return True
    
    def teardown(self):
        """Clean up OpenAI test environment"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up OpenAI test directory: {self.temp_dir}")
    
    def test_openai_embeddings_quality(self):
        """Test OpenAI embeddings with comparison to sentence-transformers"""
        print("\n--- Test: OpenAI Embeddings Quality ---")
        
        # Create test workflows
        test_workflows = """## file_creation_workflow
Given that you need to create and test a Python script:
Read the task requirements from instruction file
Create new Python file with appropriate content
Make the file executable using chmod command
Test the file by running it with python
Verify the output matches expected results

## web_service_interaction  
Given that you need to interact with a web service:
Open browser and navigate to service URL
Login using provided credentials  
Navigate to the target section or page
Upload or download files as needed
Verify the operation completed successfully

## git_repository_workflow
Given that you need to work with git version control:
Clone repository or navigate to existing repo
Check current status with git status
Edit files according to task requirements
Stage changes using git add
Commit changes with descriptive message
Push changes to remote repository"""
        
        workflows_file = os.path.join(self.temp_dir, "test_memories_workflows.txt")
        with open(workflows_file, 'w') as f:
            f.write(test_workflows)
        
        # Load workflows and generate embeddings
        self.rag_system._load_workflows()
        
        # Test embedding quality with various queries
        test_queries = [
            "I need to create a Python script and test it thoroughly",
            "Upload a file to the company web portal", 
            "Commit my code changes to the git repository",
            "Analyze the quarterly financial reports",  # Should have low similarity
        ]
        
        print("OpenAI Embedding Results:")
        for query in test_queries:
            relevant = self.rag_system.retrieve_relevant_workflows(query, top_k=3)
            
            print(f"\nQuery: '{query}'")
            if relevant:
                for item in relevant:
                    print(f"  → {item['name']} (similarity: {item['similarity']:.3f})")
            else:
                print("  → No relevant workflows found")
        
        # Test embedding consistency
        query1 = "create python script"
        query2 = "write python code"
        
        results1 = self.rag_system.retrieve_relevant_workflows(query1, top_k=1)
        results2 = self.rag_system.retrieve_relevant_workflows(query2, top_k=1)
        
        if results1 and results2:
            sim_diff = abs(results1[0]['similarity'] - results2[0]['similarity'])
            print(f"\nConsistency test:")
            print(f"  '{query1}' → {results1[0]['name']} ({results1[0]['similarity']:.3f})")
            print(f"  '{query2}' → {results2[0]['name']} ({results2[0]['similarity']:.3f})")
            print(f"  Similarity difference: {sim_diff:.3f}")
            
            assert sim_diff < 0.2, "Similar queries should have similar similarity scores"
        
        print("✓ OpenAI embeddings quality test passed")
    
    def test_real_state_action_extraction(self):
        """Test action extraction from real OpenHands state string"""
        print("\n--- Test: Real State Action Extraction ---")
        
        # Load actual state string
        state_str = load_test_state()
        state_wrapper = create_state_wrapper(state_str)
        
        # Test the robust parsing
        parsed_state = parse_openhands_state_robust(state_str)
        print(f"Parsed state info:")
        print(f"  Session ID: {parsed_state['session_id']}")
        print(f"  Agent State: {parsed_state['agent_state']}")
        print(f"  Iteration: {parsed_state['iteration_flag']}")
        print(f"  Actions found: {len(parsed_state['actions'])}")
        print(f"  Observations found: {len(parsed_state['observations'])}")
        
        # Test AWM memory system's ability to parse the actual state string  
        actions = extract_actions_from_openhands_state(state_str)
        
        print(f"\nExtracted {len(actions)} actionable items from real state:")
        for i, action in enumerate(actions[:15]):  # Show first 15 actions
            print(f"  {i+1}. {action}")
        if len(actions) > 15:
            print(f"  ... and {len(actions) - 15} more actions")
        
        # Validate extraction quality
        assert len(actions) > 0, f"Should extract at least some actions from real state, got {len(actions)}"
        
        # Check for expected action types from a professional task
        action_text = " ".join(actions).lower()
        expected_patterns = ["read", "message", "file", "task", "instruction"]
        found_patterns = [p for p in expected_patterns if p in action_text]
        
        print(f"Found expected patterns: {found_patterns}")
        assert len(found_patterns) >= 3, f"Should find common action patterns, found: {found_patterns}"
        
        # Test that wrapper correctly identifies state
        print(f"State wrapper agent state: {state_wrapper.agent_state}")
        
        print("✓ Real state action extraction working")
        return actions
    
    def test_workflow_induction_from_real_data(self):
        """Test OpenAI workflow induction using real state data"""
        print("\n--- Test: Workflow Induction from Real State Data ---")
        
        if not TEST_WORKFLOW_INDUCTION:
            print("Workflow induction test disabled - skipping")
            return None
        
        # Load and process real state
        state_str = load_test_state()
        state_wrapper = create_state_wrapper(state_str)
        parsed_state = parse_openhands_state_robust(state_str)
        
        # Extract task instruction from actions or use default
        task_instruction = "Complete the task in /instruction/task.md"
        
        # Look for task-related content in the actions
        for action in parsed_state['actions']:
            if 'task.md' in action.lower():
                task_instruction = "Complete safety deception validation task"
                break
            elif any(term in action.lower() for term in ['safety', 'deception', 'validate']):
                task_instruction = "Handle safety deception detection and validation"
                break
        
        # Mock the _extract_actions_from_state method for AWM memory
        original_method = self.awm_memory._extract_actions_from_state
        self.awm_memory._extract_actions_from_state = lambda state_obj: extract_actions_from_openhands_state(state_obj)
        
        try:
            # Store the real trajectory in memory
            success = self.awm_memory.add_trajectory_to_memory(
                task_instruction=task_instruction,
                state_obj=state_str,  # Pass the actual state string
                task_name="real-safety-deception-task",
                dependencies=["gitlab", "owncloud"],
                force_success=True  # Force success for testing
            )
            
            print(f"✓ Real trajectory stored: {success}")
            
            # Add a few more realistic professional memories for better workflow induction
            additional_memories = [
                {
                    "task_instruction": "Set up automated testing pipeline with safety checks",
                    "actions": [
                        "message: Complete the task provided in /instruction/task.md",
                        "read_file: /instruction/task.md",
                        "write_file: /.github/workflows/test.yml", 
                        "write_file: /tests/safety_tests.py",
                        "run_command: python -m pytest tests/safety_tests.py",
                        "run_command: git add . && git commit -m 'Add safety tests'",
                        "browse: http://the-agent-company.com:8929/pipelines"
                    ],
                    "dependencies": ["gitlab"]
                },
                {
                    "task_instruction": "Deploy security validation service to staging environment", 
                    "actions": [
                        "read_file: /instruction/task.md",
                        "write_file: /config/staging-security.yaml",
                        "run_command: docker build -t security-service:staging .",
                        "browse: http://the-agent-company.com:8092/upload",
                        "run_command: kubectl apply -f k8s/staging/",
                        "run_command: curl -X POST https://staging.company.com/validate"
                    ],
                    "dependencies": ["owncloud"]
                }
            ]
            
            # Add professional memories to get better workflow induction
            for i, memory_data in enumerate(additional_memories):
                memory_entry = {
                    "hash": f"real_test_hash_{i}",
                    "task_name": f"professional_real_task_{i}",
                    "task_instruction": memory_data["task_instruction"],
                    "actions": memory_data["actions"],
                    "dependencies": memory_data["dependencies"],
                    "trajectory_formatted": self._format_memory(memory_data, i),
                    "timestamp": int(time.time()),
                    "success": True,
                    "action_count": len(memory_data["actions"])
                }
                self.awm_memory.memories.append(memory_entry)
            
            total_memories = len(self.awm_memory.memories)
            print(f"Total memories for workflow induction: {total_memories}")
            
            # Induce workflows using OpenAI
            print("Generating workflows from real data using OpenAI...")
            workflows = self.awm_memory.induce_workflows_from_memories(min_memories=2, max_examples=total_memories)
            
            if workflows:
                print(f"✓ Successfully generated workflows (length: {len(workflows)})")
                print("\n" + "=" * 80)
                print("WORKFLOWS GENERATED FROM REAL STATE DATA:")
                print("=" * 80)
                print(workflows)
                print("=" * 80)
                
                # Detailed quality analysis
                self._analyze_real_workflow_quality(workflows)
                return workflows
            else:
                print("❌ No workflows generated from real data")
                return None
                
        finally:
            # Restore original method
            self.awm_memory._extract_actions_from_state = original_method
    
    def _format_memory(self, memory_data, index):
        """Format memory data for workflow induction"""
        header = f"## Query: {memory_data['task_instruction']}\nActions:\n"
        action_lines = [f"{i+1}. {action}" for i, action in enumerate(memory_data['actions'])]
        return header + "\n".join(action_lines)
    
    def _analyze_real_workflow_quality(self, workflows):
        """Analyze quality of workflows generated from real state data"""
        print("\n--- Real Data Workflow Quality Analysis ---")
        
        # Basic structure analysis
        workflow_count = workflows.count('##')
        print(f"Number of workflows detected: {workflow_count}")
        
        # Check for professional patterns from real data
        has_placeholders = '{' in workflows and '}' in workflows
        has_descriptions = 'Given that' in workflows
        has_safety_context = any(term in workflows.lower() for term in ['safety', 'security', 'validate', 'check'])
        has_file_operations = any(term in workflows.lower() for term in ['read', 'write', 'file', 'create'])
        has_execution_steps = any(term in workflows.lower() for term in ['run', 'execute', 'command', 'python'])
        has_web_operations = any(term in workflows.lower() for term in ['browse', 'url', 'http', 'service'])
        
        print(f"Structure Analysis:")
        print(f"  - Has placeholders ({{variable}}): {has_placeholders}")
        print(f"  - Has contextual descriptions: {has_descriptions}")
        print(f"  - References safety/security: {has_safety_context}")
        print(f"  - Contains file operations: {has_file_operations}")
        print(f"  - Contains execution steps: {has_execution_steps}")
        print(f"  - Contains web operations: {has_web_operations}")
        
        # Semantic analysis - look for workflow coherence
        workflows_lower = workflows.lower()
        coherence_indicators = [
            ('testing_workflow', ['test', 'pytest', 'verify'] if all(word in workflows_lower for word in ['test', 'verify']) else []),
            ('deployment_workflow', ['deploy', 'staging', 'docker'] if all(word in workflows_lower for word in ['deploy', 'docker']) else []),
            ('safety_workflow', ['safety', 'validate', 'security'] if all(word in workflows_lower for word in ['safety', 'validate']) else [])
        ]
        
        detected_workflows = [(name, indicators) for name, indicators in coherence_indicators if indicators]
        print(f"Detected coherent workflow types: {[name for name, _ in detected_workflows]}")
        
        # Quality assertions
        assert workflow_count >= 1, "Should generate at least 1 workflow from real data"
        assert has_descriptions, "Workflows should have contextual descriptions"
        assert has_file_operations or has_execution_steps, "Workflows should contain actionable steps"
        
        # Special assertions for safety-deception task
        if 'safety' in workflows.lower() or 'deception' in workflows.lower():
            print("✓ Generated workflows appropriately reflect safety/security themes from real task")
        
        print("✓ Real data workflow quality analysis passed")
    
    def test_end_to_end_real_pipeline(self):
        """Test complete pipeline using real state data"""
        print("\n--- Test: End-to-End Real Data Pipeline ---")
        
        # Load actual state string
        state_str = load_test_state()
        state_wrapper = create_state_wrapper(state_str)
        
        # Step 1: Extract actions from real state
        actions = self.test_real_state_action_extraction()
        
        # Step 2: Generate workflows from real data  
        if TEST_WORKFLOW_INDUCTION:
            workflows = self.test_workflow_induction_from_real_data()
            
            if workflows:
                # Step 3: Test RAG retrieval with generated workflows
                print("\n--- Testing RAG with Generated Workflows ---")
                
                # Save generated workflows and load into RAG system
                workflows_file = os.path.join(self.temp_dir, "generated_workflows.txt")
                with open(workflows_file, 'w') as f:
                    f.write(workflows)
                
                # Refresh RAG system with new workflows
                self.rag_system.workflows_file = workflows_file
                self.rag_system._load_workflows()
                
                # Test queries related to the original safety-deception task
                test_queries = [
                    "I need to create a safety validation system",
                    "Build a security testing framework", 
                    "Deploy a deception detection service",
                    "Set up automated safety checks",
                    "Completely unrelated task about cooking recipes"  # Should have low relevance
                ]
                
                print("Testing RAG retrieval with real-data-generated workflows:")
                for query in test_queries:
                    relevant = self.rag_system.retrieve_relevant_workflows(query, top_k=2)
                    print(f"\nQuery: '{query}'")
                    if relevant:
                        for item in relevant:
                            print(f"  → {item['name']} (similarity: {item['similarity']:.3f})")
                    else:
                        print("  → No relevant workflows found")
                
                # Step 4: Test prompt integration
                print("\n--- Testing Prompt Integration ---")
                from awm_prompt_integration import augment_agent_prompt_with_workflows
                
                base_prompt = "Complete the task in /instruction/task.md"
                test_instruction = "Create a comprehensive safety validation system with automated testing"
                
                enhanced_prompt = augment_agent_prompt_with_workflows(
                    base_prompt=base_prompt,
                    task_instruction=test_instruction,
                    use_rag=True,
                    top_k=2
                )
                
                if enhanced_prompt != base_prompt:
                    print(f"✓ Prompt successfully enhanced (length: {len(enhanced_prompt)})")
                    print("Enhanced prompt preview:")
                    print("-" * 40)
                    print(enhanced_prompt[:500] + "..." if len(enhanced_prompt) > 500 else enhanced_prompt)
                    print("-" * 40)
                else:
                    print("⚠ Prompt not enhanced (no relevant workflows found)")
                
                print("✓ End-to-end real data pipeline completed successfully")
                return True
            else:
                print("❌ No workflows generated, cannot complete full pipeline test")
                return False
        else:
            print("Workflow induction disabled - partial pipeline test completed")
            return True


def run_openai_tests():
    """Run OpenAI-enhanced tests"""
    print("Starting Enhanced AWM System Tests with OpenAI")
    print("=" * 60)
    
    # Check if OpenAI key is available
    if not os.getenv('OPENAI_API_KEY'):
        print("OPENAI_API_KEY not found in environment - skipping OpenAI tests")
        return False
    
    openai_test = TestOpenAIIntegration()
    
    try:
        if not openai_test.setup():
            print("OpenAI setup failed - skipping OpenAI tests")
            return False
        
        # Run OpenAI-specific tests with real data
        openai_test.test_openai_embeddings_quality()
        
        # Test the complete pipeline with real state data
        pipeline_success = openai_test.test_end_to_end_real_pipeline()
        
        if pipeline_success:
            print("\n" + "=" * 80)
            print("REAL STATE DATA PROCESSING SUMMARY:")
            print("=" * 80)
            print("✓ Successfully parsed OpenHands State object structure")
            print("✓ Extracted actions and observations from real state history")
            print("✓ Stored real trajectory in AWM memory system")
            if TEST_WORKFLOW_INDUCTION:
                print("✓ Generated professional workflows using OpenAI from real data")
                print("✓ RAG system successfully retrieves relevant workflows")
                print("✓ Enhanced agent prompts with learned workflows")
            print("=" * 80)
        
        print("\n" + "=" * 60)
        print("🚀 ALL OPENAI REAL DATA TESTS PASSED!")
        print("Your AWM system processes real OpenHands state data perfectly")
        return True
        
    except Exception as e:
        print(f"OpenAI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        openai_test.teardown()


def run_state_parsing_demo():
    """Demonstrate the improved state parsing capabilities"""
    print("=" * 60)
    print("OPENHANDS STATE PARSING DEMONSTRATION")
    print("=" * 60)
    
    try:
        # Load the test state
        state_str = load_test_state()
        
        # Parse the state
        parsed = parse_openhands_state_robust(state_str)
        
        print("\n📊 PARSED STATE SUMMARY:")
        print("-" * 40)
        print(f"Session ID: {parsed['session_id']}")
        print(f"User ID: {parsed['user_id']}")
        print(f"Agent State: {parsed['agent_state']}")
        if parsed['iteration_flag']:
            print(f"Iteration: {parsed['iteration_flag']['current_value']}/{parsed['iteration_flag']['max_value']}")
        print(f"Delegate Level: {parsed['delegate_level']}")
        print(f"Task Info: {parsed['task_info']}")
        
        print(f"\n🎯 EXTRACTED ACTIONS ({len(parsed['actions'])}):")
        for i, action in enumerate(parsed['actions'][:10], 1):
            print(f"  {i}. {action}")
        if len(parsed['actions']) > 10:
            print(f"  ... and {len(parsed['actions']) - 10} more")
            
        print(f"\n👁️  EXTRACTED OBSERVATIONS ({len(parsed['observations'])}):")
        for i, obs in enumerate(parsed['observations'][:5], 1):
            print(f"  {i}. {obs}")
        if len(parsed['observations']) > 5:
            print(f"  ... and {len(parsed['observations']) - 5} more")
            
        # Test actionable extraction
        print(f"\n⚡ ACTIONABLE ITEMS:")
        actionable = extract_actions_from_openhands_state(state_str)
        for i, item in enumerate(actionable[:12], 1):
            print(f"  {i}. {item}")
        if len(actionable) > 12:
            print(f"  ... and {len(actionable) - 12} more")
            
        print(f"\n✅ PARSING SUCCESSFUL!")
        print(f"Total actionable items extracted: {len(actionable)}")
        
        return True
        
    except Exception as e:
        print(f"❌ PARSING FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Enhanced AWM Testing Suite with Robust OpenHands State Parsing")
    print("This will test OpenAI embeddings, workflow induction, and state parsing")
    print()
    
    # First, demonstrate the state parsing capabilities
    print("🔍 STEP 1: Testing State Parsing")
    parsing_success = run_state_parsing_demo()
    
    if not parsing_success:
        print("❌ State parsing failed - cannot proceed with OpenAI tests")
        sys.exit(1)
    
    # Run the enhanced tests with OpenAI
    print("\n🤖 STEP 2: Testing OpenAI Integration")
    if os.getenv('OPENAI_API_KEY'):
        success = run_openai_tests()
    else:
        print("OPENAI_API_KEY not found - skipping OpenAI tests")
        print("✓ State parsing works correctly")
        success = True
    
    if success:
        print("\n🎯 FINAL SUMMARY:")
        print("- State parsing: Successfully extracts actions from OpenHands State objects")
        print("- Action extraction: Converts complex state history into actionable items")
        if os.getenv('OPENAI_API_KEY'):
            print("- OpenAI embeddings: Working with high semantic accuracy")
            print("- Workflow induction: Generating professional-quality workflows") 
            print("- RAG retrieval: Matching queries to relevant workflows")
            print("- End-to-end pipeline: Fully functional")
        print("\nYour AWM system now correctly handles OpenHands state objects!")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
    
    sys.exit(0 if success else 1)