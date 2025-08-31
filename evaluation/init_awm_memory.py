import os
import sys
from pathlib import Path

# Add evaluation directory to path
sys.path.append(str(Path(__file__).parent))

from awm_memory import AWMMemorySystem

def main():
    # Initialize memory system
    memory_file = os.path.join(os.path.dirname(__file__), "awm_memories.json")
    awm = AWMMemorySystem(memory_file=memory_file)
    
    print("AWM Memory System Initialized")
    print(f"Memory file: {memory_file}")
    
    # Get and display current stats
    stats = awm.get_memory_stats()
    print(f"\nCurrent Memory Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Check for existing workflows
    workflows_file = memory_file.replace('.json', '_workflows.txt')
    if os.path.exists(workflows_file):
        print(f"\nExisting workflows file: {workflows_file}")
        with open(workflows_file, 'r') as f:
            content = f.read()
            print(f"Workflows preview (first 500 chars):\n{content[:500]}...")
    else:
        print(f"\nNo existing workflows file found. Will be created at: {workflows_file}")
    
    print("\nAWM Memory system is ready!")

if __name__ == "__main__":
    main()