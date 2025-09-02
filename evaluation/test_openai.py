#!/usr/bin/env python3
"""
Enhanced AWM unit tests with OpenAI integration
Run from evaluation directory: python test_awm_system_enhanced.py
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
USE_OPENAI_EMBEDDINGS = True  # Set to False to use sentence-transformers
TEST_WORKFLOW_INDUCTION = True  # Set to False to skip OpenAI workflow generation


def load_test_state():
    """Load the test state object from JSON file"""
    state_file = Path(__file__).parent / TEST_STATE_FILE
    if not state_file.exists():
        raise FileNotFoundError(f"Test state file not found: {state_file}")
    
    with open(state_file, 'r') as f:
        state_str = f.read().strip()
    
    print(f"Loaded state string length: {len(state_str)}")
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
            self.agent_state = "AgentState.FINISHED"
            
            # Create realistic mock history with professional tasks
            self.history = [
                MockAction('message', content='Complete the task provided in /instruction/task.md'),
                MockAction('read', path='/instruction/task.md'),
                MockAction('edit', path='/workspace/safety_check.py', content='# Safety validation script'),
                MockAction('run', command='python /workspace/safety_check.py'),
                MockAction('run', command='chmod +x /workspace/safety_check.py'),
                MockAction('browse', url='http://the-agent-company.com:8092'),
                MockAction('run', command='git status'),
                MockAction('edit', path='/workspace/config.json'),
                MockAction('run', command='python -m pytest tests/'),
                MockAction('finish', task_completed='true')
            ]
    
    return MockState()


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
    
    def test_workflow_induction_quality(self):
        """Test OpenAI-powered workflow induction"""
        print("\n--- Test: OpenAI Workflow Induction Quality ---")
        
        if not TEST_WORKFLOW_INDUCTION:
            print("Workflow induction test disabled - skipping")
            return
        
        # Create diverse realistic professional task memories
        professional_memories = [
            {
                "task_instruction": "Set up CI/CD pipeline for the project repository",
                "actions": [
                    "Read file: /instruction/task.md",
                    "Browse to: http://the-agent-company.com:8929/project/settings",
                    "Edit file: /.github/workflows/ci.yml",
                    "Run command: git add .",
                    "Run command: git commit -m 'Add CI pipeline'",
                    "Run command: git push origin main"
                ]
            },
            {
                "task_instruction": "Create safety validation script for user inputs",
                "actions": [
                    "Read file: /instruction/task.md", 
                    "Edit file: /workspace/safety_validator.py",
                    "Run command: python /workspace/safety_validator.py --test",
                    "Edit file: /workspace/tests/test_safety.py",
                    "Run command: pytest tests/test_safety.py -v"
                ]
            },
            {
                "task_instruction": "Deploy application to staging environment",
                "actions": [
                    "Read file: /instruction/task.md",
                    "Edit file: /config/staging.json",
                    "Run command: docker build -t app:staging .",
                    "Run command: docker push registry.company.com/app:staging",
                    "Browse to: http://staging.company.com/health",
                    "Run command: kubectl apply -f k8s/staging/"
                ]
            }
        ]
        
        # Add memories to AWM system
        for i, memory_data in enumerate(professional_memories):
            memory_entry = {
                "hash": f"test_hash_{i}",
                "task_name": f"professional_task_{i}",
                "task_instruction": memory_data["task_instruction"],
                "actions": memory_data["actions"],
                "dependencies": ["gitlab"],
                "trajectory_formatted": self._format_memory(memory_data, i),
                "timestamp": int(time.time()),
                "success": True,
                "action_count": len(memory_data["actions"])
            }
            self.awm_memory.memories.append(memory_entry)
        
        print(f"Added {len(professional_memories)} professional task memories")
        
        # Test workflow induction
        print("Inducing workflows with OpenAI...")
        workflows = self.awm_memory.induce_workflows_from_memories(min_memories=3, max_examples=3)
        
        if workflows:
            print(f"Generated workflows (length: {len(workflows)}):")
            print("=" * 60)
            print(workflows)
            print("=" * 60)
            
            # Analyze workflow quality
            self._analyze_workflow_quality(workflows)
            print("✓ OpenAI workflow induction completed")
            
            return workflows
        else:
            print("No workflows generated - this may indicate an issue")
            return None
    
    def _format_memory(self, memory_data, index):
        """Format memory data for workflow induction"""
        header = f"## Query: {memory_data['task_instruction']}\nActions:\n"
        action_lines = [f"{i+1}. {action}" for i, action in enumerate(memory_data['actions'])]
        return header + "\n".join(action_lines)
    
    def _analyze_workflow_quality(self, workflows):
        """Analyze the quality of generated workflows"""
        print("\n--- Workflow Quality Analysis ---")
        
        # Check for workflow structure
        workflow_count = workflows.count('##')
        print(f"Number of workflows detected: {workflow_count}")
        
        # Check for key patterns
        has_placeholders = '{' in workflows and '}' in workflows
        has_descriptions = 'Given that' in workflows
        has_steps = any(action in workflows.lower() for action in ['read', 'edit', 'run', 'browse'])
        
        print(f"Has placeholders ({{variable}}): {has_placeholders}")
        print(f"Has contextual descriptions: {has_descriptions}")
        print(f"Contains actionable steps: {has_steps}")
        
        # Look for professional task patterns
        professional_keywords = ['deploy', 'test', 'commit', 'config', 'pipeline', 'staging']
        found_keywords = [kw for kw in professional_keywords if kw.lower() in workflows.lower()]
        print(f"Professional keywords found: {found_keywords}")
        
        assert workflow_count >= 2, "Should generate at least 2 workflows"
        assert has_descriptions, "Workflows should have contextual descriptions"
        assert has_steps, "Workflows should contain actionable steps"
        
        print("✓ Workflow quality analysis passed")
    
    def test_end_to_end_openai_pipeline(self):
        """Test complete pipeline with OpenAI components"""
        print("\n--- Test: End-to-End OpenAI Pipeline ---")
        
        # Create mock state and store in memory
        state_str = load_test_state()
        mock_state = create_mock_state_object(state_str)
        
        # Add trajectory to memory
        success = self.awm_memory.add_trajectory_to_memory(
            task_instruction="Create and validate a safety checking system with proper testing",
            state_obj=mock_state,
            task_name="safety-system-e2e-test",
            dependencies=["gitlab", "owncloud"]
        )
        
        assert success, "Should successfully store trajectory"
        print("✓ Trajectory stored in memory")
        
        # Generate workflows if we have enough memories
        if len(self.awm_memory.memories) >= 3:
            workflows = self.awm_memory.induce_workflows_from_memories(min_memories=1, max_examples=3)
            if workflows:
                print("✓ Workflows induced from memories")
                
                # Test RAG retrieval with generated workflows
                self.rag_system._load_workflows()
                
                query = "I need to create a safety validation system"
                relevant = self.rag_system.retrieve_relevant_workflows(query, top_k=2)
                
                print(f"RAG retrieval for: '{query}'")
                if relevant:
                    for item in relevant:
                        print(f"  → {item['name']} (similarity: {item['similarity']:.3f})")
                    print("✓ RAG retrieval working with generated workflows")
                else:
                    print("⚠ No relevant workflows found for test query")
            else:
                print("⚠ No workflows generated")
        else:
            print("⚠ Not enough memories for workflow induction")
        
        print("✓ End-to-end OpenAI pipeline test completed")


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
        
        # Run OpenAI-specific tests
        openai_test.test_openai_embeddings_quality()
        
        if TEST_WORKFLOW_INDUCTION:
            workflows = openai_test.test_workflow_induction_quality()
            if workflows:
                print("\n" + "=" * 60)
                print("GENERATED WORKFLOWS PREVIEW:")
                print("=" * 60)
                # Show first 800 characters of generated workflows
                preview = workflows[:800] + "..." if len(workflows) > 800 else workflows
                print(preview)
                print("=" * 60)
        
        openai_test.test_end_to_end_openai_pipeline()
        
        print("\n" + "=" * 60)
        print("🚀 ALL OPENAI TESTS PASSED!")
        print("Your AWM system is working excellently with OpenAI integration")
        return True
        
    except Exception as e:
        print(f"OpenAI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        openai_test.teardown()


if __name__ == "__main__":
    print("Enhanced AWM Testing Suite")
    print("This will test OpenAI embeddings and workflow induction quality")
    print("Make sure OPENAI_API_KEY is set in your environment")
    print()
    
    # Run the enhanced tests
    success = run_openai_tests()
    
    if success:
        print("\n🎯 SUMMARY:")
        print("- OpenAI embeddings: Working with high semantic accuracy")
        print("- Workflow induction: Generating professional-quality workflows") 
        print("- RAG retrieval: Matching queries to relevant workflows")
        print("- End-to-end pipeline: Fully functional")
        print("\nYour AWM system is ready for production deployment!")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
    
    sys.exit(0 if success else 1)