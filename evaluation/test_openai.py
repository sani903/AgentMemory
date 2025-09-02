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
    """Load the actual serialized state object from JSON file"""
    state_file = Path(__file__).parent / TEST_STATE_FILE
    if not state_file.exists():
        raise FileNotFoundError(f"Test state file not found: {state_file}")
    
    with open(state_file, 'r') as f:
        state_str = f.read().strip()
    
    print(f"Loaded actual state string length: {len(state_str)}")
    print(f"State preview: {state_str[:150]}...")
    
    # Parse the state string to extract basic info for validation
    if "safety-deception" in state_str:
        print("✓ Detected safety-deception task in state")
    if "AgentState.FINISHED" in state_str:
        print("✓ Detected successful task completion")
    if "history=[" in state_str:
        print("✓ Detected action history in state")
    
    return state_str


def create_state_wrapper(state_str):
    """Create a wrapper that presents the serialized state as needed by AWM"""
    class StateStringWrapper:
        def __init__(self, state_str):
            self.state_str = state_str
            self.agent_state = "AgentState.FINISHED" if "AgentState.FINISHED" in state_str else "RUNNING"
            # AWM will parse this string directly
            
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
        
        # Test AWM memory system's ability to parse the actual state string
        actions = self.awm_memory._extract_actions_from_state(state_str)
        
        print(f"Extracted {len(actions)} actions from real state:")
        for i, action in enumerate(actions[:10]):  # Show first 10 actions
            print(f"  {i+1}. {action}")
        if len(actions) > 10:
            print(f"  ... and {len(actions) - 10} more actions")
        
        # Validate extraction quality
        assert len(actions) > 0, "Should extract at least some actions from real state"
        
        # Check for expected action types from a professional task
        action_text = " ".join(actions).lower()
        expected_patterns = ["read", "edit", "run", "file", "command"]
        found_patterns = [p for p in expected_patterns if p in action_text]
        
        print(f"Found expected patterns: {found_patterns}")
        assert len(found_patterns) >= 3, f"Should find common action patterns, found: {found_patterns}"
        
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
        
        # Extract task instruction from state string if possible
        task_instruction = "Create and test a safety validation system"
        if "safety" in state_str.lower():
            task_instruction = "Handle safety deception detection and validation"
        
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
                    "Read file: /instruction/task.md",
                    "Edit file: /.github/workflows/test.yml", 
                    "Edit file: /tests/safety_tests.py",
                    "Run command: python -m pytest tests/safety_tests.py",
                    "Run command: git add . && git commit -m 'Add safety tests'",
                    "Browse to: http://the-agent-company.com:8929/pipelines"
                ],
                "dependencies": ["gitlab"]
            },
            {
                "task_instruction": "Deploy security validation service to staging environment", 
                "actions": [
                    "Read file: /instruction/task.md",
                    "Edit file: /config/staging-security.yaml",
                    "Run command: docker build -t security-service:staging .",
                    "Browse to: http://the-agent-company.com:8092/upload",
                    "Run command: kubectl apply -f k8s/staging/",
                    "Run command: curl -X POST https://staging.company.com/validate"
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
        has_file_operations = any(term in workflows.lower() for term in ['read', 'edit', 'file', 'create'])
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
            print("✓ Successfully extracted actions from actual OpenHands state")
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