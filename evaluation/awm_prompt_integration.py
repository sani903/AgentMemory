"""
AWM Prompt Integration for TheAgentCompany
This provides functions to integrate AWM workflows into agent prompts
"""

import os
from typing import Optional
try:
    from awm_memory import awm_memory
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

def get_relevant_workflows_for_task(task_instruction: str, use_rag: bool = True, top_k: int = 3) -> str:
    if not MEMORY_AVAILABLE:
        return ""
    return awm_memory.get_relevant_workflows(task_instruction, use_rag=use_rag, top_k=top_k)

def augment_agent_prompt_with_workflows(base_prompt: str, task_instruction: str, 
                                       use_rag: bool = True, top_k: int = 3) -> str:
    """Augment the base agent prompt with relevant workflows using RAG"""
    workflows = get_relevant_workflows_for_task(task_instruction, use_rag=use_rag, top_k=top_k)
    
    if not workflows.strip():
        return base_prompt
    
    # RAG workflows already include formatting, so use them directly
    workflow_section = f"""

{workflows}

## Current Task

"""
    
    # Insert workflows section before the current task
    if "Complete the task in /instruction/task.md" in base_prompt:
        return base_prompt.replace(
            "Complete the task in /instruction/task.md",
            workflow_section + "Complete the task in /instruction/task.md"
        )
    else:
        return workflow_section + base_prompt

def create_workflow_system_message() -> str:
    """Create a system message that explains how to use workflows"""
    return """You are an AI assistant that can leverage learned workflows from previous successful task executions. 

When you see workflow patterns that match your current task, you can:
1. Identify which workflow components are relevant
2. Adapt the workflow steps to your specific task context
3. Use the workflow as a guide while remaining flexible for task-specific requirements

Workflows are templates with placeholders like {variable-name} that you should replace with task-specific values."""

# Example usage in OpenHands integration (for future use):
def modify_agent_instruction_with_workflows(instruction: str, task_instruction: str) -> str:
    """Modify agent instruction to include relevant workflows"""
    workflows = get_relevant_workflows_for_task(task_instruction)
    
    if workflows.strip():
        workflow_augmented = f"""{instruction}

AVAILABLE LEARNED WORKFLOWS:
{workflows}

You can reference these workflows when they are relevant to your current task. Adapt them as needed for the specific requirements.
"""
        return workflow_augmented
    
    return instruction