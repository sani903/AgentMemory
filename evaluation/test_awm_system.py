#!/usr/bin/env python3
"""
Unit tests for AWM system components
Run from evaluation directory: python test_awm_system.py
"""

import os
import sys
import json
import tempfile
import shutil
import time
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test configuration
TEST_STATE_FILE = "state_safety-deception.json"
USE_SENTENCE_TRANSFORMERS = True  # Set to False to test keyword fallback

# Make this global for access in test classes
global USE_SENTENCE_TRANSFORMERS


def load_test_state():
    """Load the test state object from JSON file"""
    state_file = Path(__file__).parent / TEST_STATE_FILE
    if not state_file.exists():
        raise FileNotFoundError(f"Test state file not found: {state_file}")
    
    with open(state_file, 'r') as f:
        state_str = f.read().strip()
    
    print(f"Loaded state string length: {len(state_str)}")
    print(f"State preview: {state_str[:200]}...")
    return state_str


def create_mock_state_object(state_str):
    """Create a mock state object that behaves like OpenHands State"""
    class MockAction:
        def __init__(self, action_type, **kwargs):
            self.action = action_type
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class MockState:
        def __init__(self):
            # Parse some basic actions from the state string
            self.agent_state = "AgentState.FINISHED" if "finish" in state_str.lower() else "RUNNING"
            
            # Create mock history with various action types
            self.history = [
                MockAction('message', content='Complete the task provided in /instruction/task.md'),
                MockAction('read', path='/instruction/task.md'),
                MockAction('edit', path='/workspace/safety_check.py'),
                MockAction('run', command='python /workspace/safety_check.py'),
                MockAction('browse', url='http://example.com'),
                MockAction('finish', task_completed='true')
            ]
    
    return MockState()


class TestAWMMemorySystem:
    """Test AWM Memory System functionality"""
    
    def __init__(self):
        self.temp_dir = None
        self.awm_memory = None
    
    def setup(self):
        """Set up test environment"""
        print("\n=== Setting Up Test Environment ===")
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp(prefix="awm_test_")
        print(f"Test directory: {self.temp_dir}")
        
        # Import and initialize AWM memory with test directory
        from awm_memory import AWMMemorySystem
        test_memory_file = os.path.join(self.temp_dir, "test_memories.json")
        self.awm_memory = AWMMemorySystem(memory_file=test_memory_file, model="gpt-3.5-turbo")
        print("AWM Memory System initialized")
    
    def teardown(self):
        """Clean up test environment"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up test directory: {self.temp_dir}")
    
    def test_action_extraction(self, mock_state):
        """Test action extraction from state object"""
        print("\n--- Test: Action Extraction ---")
        
        actions = self.awm_memory._extract_actions_from_state(mock_state)
        
        print(f"Extracted {len(actions)} actions:")
        for i, action in enumerate(actions):
            print(f"  {i+1}. {action}")
        
        assert len(actions) > 0, "Should extract at least some actions"
        assert any("read" in action.lower() for action in actions), "Should extract read actions"
        print("✓ Action extraction working")
        return actions
    
    def test_memory_storage(self, mock_state):
        """Test storing trajectory in memory"""
        print("\n--- Test: Memory Storage ---")
        
        # Test successful storage
        success = self.awm_memory.add_trajectory_to_memory(
            task_instruction="Test task: Check safety deception handling",
            state_obj=mock_state,
            task_name="safety-deception-test",
            dependencies=["gitlab", "owncloud"]
        )
        
        assert success, "Memory storage should succeed"
        assert len(self.awm_memory.memories) == 1, "Should have 1 memory stored"
        
        memory = self.awm_memory.memories[0]
        assert memory["task_name"] == "safety-deception-test"
        assert len(memory["actions"]) > 0
        assert "gitlab" in memory["dependencies"]
        
        print("✓ Memory storage working")
        
        # Test duplicate detection
        success2 = self.awm_memory.add_trajectory_to_memory(
            task_instruction="Test task: Check safety deception handling",
            state_obj=mock_state,
            task_name="safety-deception-test-duplicate",
            dependencies=["gitlab"]
        )
        
        assert not success2, "Duplicate should not be stored"
        assert len(self.awm_memory.memories) == 1, "Should still have only 1 memory"
        print("✓ Duplicate detection working")
        
        return memory
    
    def test_memory_persistence(self):
        """Test memory file saving and loading"""
        print("\n--- Test: Memory Persistence ---")
        
        # Check file was created
        assert os.path.exists(self.awm_memory.memory_file), "Memory file should exist"
        
        # Create new instance and check it loads the memory
        from awm_memory import AWMMemorySystem
        awm_memory2 = AWMMemorySystem(memory_file=self.awm_memory.memory_file)
        
        assert len(awm_memory2.memories) == 1, "New instance should load existing memories"
        assert awm_memory2.memories[0]["task_name"] == "safety-deception-test"
        
        print("✓ Memory persistence working")
    
    def test_workflow_induction_structure(self):
        """Test workflow induction prompt structure (without API call)"""
        print("\n--- Test: Workflow Induction Structure ---")
        
        # Add more memories to trigger workflow induction
        mock_actions = [
            ["Read file: /instruction/task.md", "Edit file: /workspace/test.py", "Run command: python test.py"],
            ["Browse to: http://example.com", "Read file: /config/settings.json", "Edit file: /config/settings.json"],
            ["Run command: git clone repo", "Edit file: README.md", "Run command: git commit -m 'update'"]
        ]
        
        for i, actions in enumerate(mock_actions):
            memory_entry = {
                "hash": f"test_hash_{i}",
                "task_name": f"test_task_{i}",
                "task_instruction": f"Test instruction {i}",
                "actions": actions,
                "dependencies": ["gitlab"],
                "trajectory_formatted": f"## Query: Test instruction {i}\nActions:\n" + "\n".join(f"{j+1}. {a}" for j, a in enumerate(actions)),
                "timestamp": int(time.time()),
                "success": True,
                "action_count": len(actions)
            }
            self.awm_memory.memories.append(memory_entry)
        
        # Test prompt construction (without calling OpenAI)
        recent_memories = self.awm_memory.memories[-3:]
        formatted_examples = [mem["trajectory_formatted"] for mem in recent_memories]
        examples_text = "\n\n".join(formatted_examples)
        prompt = f"{self.awm_memory.instruction}\n\n{self.awm_memory.one_shot_example}\n\n{examples_text}\n\nSummary Workflows:"
        
        assert len(prompt) > 1000, "Prompt should be substantial"
        assert "Summary Workflows:" in prompt, "Prompt should have correct structure"
        assert "Test instruction" in prompt, "Prompt should include test data"
        
        print(f"Generated prompt length: {len(prompt)}")
        print(f"Prompt preview: {prompt[:300]}...")
        print("✓ Workflow induction structure working")


class TestRAGSystem:
    """Test RAG system functionality"""
    
    def __init__(self):
        self.temp_dir = None
        self.rag_system = None
    
    def setup(self):
        """Set up RAG test environment"""
        print("\n=== Setting Up RAG Test Environment ===")
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="rag_test_")
        print(f"RAG test directory: {self.temp_dir}")
        
        # Create test workflows file
        test_workflows = """## create_and_test_file
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
Verify upload was successful

## git_workflow
Given that you need to work with git repository:
Clone or navigate to repository
Read existing files to understand structure
Edit files as needed: {file-paths}
Commit changes with descriptive message
Push changes to remote repository"""
        
        workflows_file = os.path.join(self.temp_dir, "test_memories_workflows.txt")
        with open(workflows_file, 'w') as f:
            f.write(test_workflows)
        
        print(f"Created test workflows file: {workflows_file}")
        
        # Initialize RAG system
        use_sentence_transformers = USE_SENTENCE_TRANSFORMERS
        if use_sentence_transformers:
            try:
                from awm_rag_system import ProductionRAGSystem
                memory_file = os.path.join(self.temp_dir, "test_memories.json")
                self.rag_system = ProductionRAGSystem(
                    memory_file=memory_file,
                    use_openai_embeddings=False
                )
                print("RAG system initialized with sentence transformers")
            except ImportError:
                print("Sentence transformers not available, testing keyword fallback")
                use_sentence_transformers = False
        
        if not use_sentence_transformers:
            from awm_rag_system import ProductionRAGSystem
            memory_file = os.path.join(self.temp_dir, "test_memories.json")
            # Force keyword fallback
            self.rag_system = ProductionRAGSystem(memory_file=memory_file)
            self.rag_system.embedding_model = "keyword_fallback"
            self.rag_system._load_workflows()
            print("RAG system initialized with keyword fallback")
    
    def teardown(self):
        """Clean up RAG test environment"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up RAG test directory: {self.temp_dir}")
    
    def test_workflow_parsing(self):
        """Test workflow parsing from text file"""
        print("\n--- Test: Workflow Parsing ---")
        
        workflows = self.rag_system.workflows_data
        
        assert len(workflows) == 3, f"Should parse 3 workflows, got {len(workflows)}"
        
        workflow_names = [w['name'] for w in workflows]
        expected_names = ['create_and_test_file', 'upload_file_to_service', 'git_workflow']
        
        for expected in expected_names:
            assert expected in workflow_names, f"Should parse workflow: {expected}"
        
        # Check workflow structure
        first_workflow = workflows[0]
        assert 'name' in first_workflow
        assert 'description' in first_workflow
        assert 'steps' in first_workflow
        assert 'full_text' in first_workflow
        
        print(f"Parsed workflows: {workflow_names}")
        print("✓ Workflow parsing working")
    
    def test_embedding_generation(self):
        """Test embedding generation and caching"""
        print("\n--- Test: Embedding Generation ---")
        
        if self.rag_system.embedding_model == "keyword_fallback":
            print("Using keyword fallback - skipping embedding test")
            return
        
        # Check embeddings were generated
        assert len(self.rag_system.workflow_embeddings) > 0, "Should generate embeddings"
        
        # Test embedding cache
        cache_file = self.rag_system.embedding_cache_file
        assert os.path.exists(cache_file), "Embedding cache file should exist"
        
        # Test cache loading
        old_embeddings = dict(self.rag_system.workflow_embeddings)
        self.rag_system._load_embeddings_cache()
        
        assert len(self.rag_system.workflow_embeddings) == len(old_embeddings), "Cache should load correctly"
        
        print(f"Generated {len(self.rag_system.workflow_embeddings)} embeddings")
        print("✓ Embedding generation and caching working")
    
    def test_similarity_search(self):
        """Test similarity-based workflow retrieval"""
        print("\n--- Test: Similarity Search ---")
        
        test_queries = [
            "I need to create a Python script and test it",
            "Upload a document to a web service",
            "Work with git repository and commit changes",
            "Something completely unrelated to existing workflows"
        ]
        
        for query in test_queries:
            relevant = self.rag_system.retrieve_relevant_workflows(query, top_k=2)
            
            print(f"\nQuery: {query}")
            print(f"Found {len(relevant)} relevant workflows:")
            
            for item in relevant:
                print(f"  - {item['name']} (similarity: {item['similarity']:.3f})")
            
            if "Python script" in query:
                # Should find the file creation workflow
                assert len(relevant) > 0, "Should find relevant workflows for Python script"
                assert any("create" in item['name'] for item in relevant), "Should find create workflow"
            
            elif "git" in query:
                # Should find git workflow
                assert any("git" in item['name'] for item in relevant), "Should find git workflow"
        
        print("✓ Similarity search working")
    
    def test_prompt_formatting(self):
        """Test workflow formatting for agent prompts"""
        print("\n--- Test: Prompt Formatting ---")
        
        query = "Create and test a Python file"
        relevant = self.rag_system.retrieve_relevant_workflows(query, top_k=2)
        formatted = self.rag_system.format_workflows_for_prompt(relevant)
        
        if relevant:
            assert "## Relevant Learned Workflows" in formatted, "Should have proper header"
            assert "relevance:" in formatted, "Should show relevance scores"
            assert len(formatted) > 100, "Formatted output should be substantial"
            
            print(f"Formatted prompt length: {len(formatted)}")
            print(f"Formatted preview:\n{formatted[:400]}...")
        else:
            print("No relevant workflows found for test query")
        
        print("✓ Prompt formatting working")


class TestPromptIntegration:
    """Test prompt integration functionality"""
    
    def test_prompt_augmentation(self):
        """Test augmenting agent prompts with workflows"""
        print("\n--- Test: Prompt Augmentation ---")
        
        try:
            from awm_prompt_integration import augment_agent_prompt_with_workflows
            
            base_prompt = "Complete the task in /instruction/task.md"
            task_instruction = "Create a Python script to process data"
            
            # Test without RAG (should return original if no workflows)
            enhanced_prompt = augment_agent_prompt_with_workflows(
                base_prompt=base_prompt,
                task_instruction=task_instruction,
                use_rag=False
            )
            
            print(f"Base prompt: {base_prompt}")
            print(f"Enhanced prompt length: {len(enhanced_prompt)}")
            
            if enhanced_prompt != base_prompt:
                print("✓ Prompt augmentation working (found workflows)")
                assert "workflow" in enhanced_prompt.lower(), "Enhanced prompt should contain workflow references"
            else:
                print("✓ Prompt augmentation working (no workflows found, returned original)")
        
        except ImportError:
            print("Prompt integration module not available - skipping test")


def run_all_tests():
    """Run all tests"""
    print("Starting AWM System Unit Tests")
    print("=" * 50)
    
    # Load test data
    try:
        state_str = load_test_state()
        mock_state = create_mock_state_object(state_str)
        print("✓ Test data loaded successfully")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return False
    
    # Test AWM Memory System
    memory_test = TestAWMMemorySystem()
    try:
        memory_test.setup()
        
        actions = memory_test.test_action_extraction(mock_state)
        memory = memory_test.test_memory_storage(mock_state)
        memory_test.test_memory_persistence()
        memory_test.test_workflow_induction_structure()
        
        print("\n✓ All AWM Memory System tests passed")
        
    except Exception as e:
        print(f"AWM Memory System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        memory_test.teardown()
    
    # Test RAG System
    rag_test = TestRAGSystem()
    try:
        rag_test.setup()
        
        rag_test.test_workflow_parsing()
        rag_test.test_embedding_generation()
        rag_test.test_similarity_search()
        rag_test.test_prompt_formatting()
        
        print("\n✓ All RAG System tests passed")
        
    except Exception as e:
        print(f"RAG System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        rag_test.teardown()
    
    # Test Prompt Integration
    prompt_test = TestPromptIntegration()
    try:
        prompt_test.test_prompt_augmentation()
        print("\n✓ Prompt Integration test passed")
    except Exception as e:
        print(f"Prompt Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED!")
    print("AWM system is working correctly")
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)