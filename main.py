import yaml
import json
import os
from crewai import Agent, Task, Crew
from gemini_llm import GeminiLLM

MEMORY_FILE = "agent_memory.json"

def load_yaml(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_agent_memory(agent_names):
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            memory = json.load(f)
    else:
        memory = {}
    for name in agent_names:
        if name not in memory:
            memory[name] = []
    return memory

def save_agent_memory(memory):
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(memory, f, indent=2)

def build_agents(agent_defs, llm):
    agents = {}
    for agent_def in agent_defs:
        agent = Agent(
            role=agent_def['name'],
            goal=agent_def['goal'],
            backstory=agent_def.get('role', ''),
            llm=llm
        )
        agents[agent_def['name']] = agent
    return agents

def build_tasks(task_defs, agents):
    tasks = []
    for task_def in task_defs:
        tasks.append(Task(
            description=task_def['description'],
            agent=agents[task_def['agent']],
            expected_output=task_def.get('output', '')
        ))
    return tasks

def run_crewai_workflow(agents_yaml, tasks_yaml, code_snippet, error_message):
    agent_defs = load_yaml(agents_yaml)['agents']
    task_defs = load_yaml(tasks_yaml)['tasks']
    agent_names = [a['name'] for a in agent_defs]
    llm = GeminiLLM(api_key="AIzaSyB6Tq_t044mnyWJOhr5YqFXhIGJf858VQI")
    memory = load_agent_memory(agent_names)
    agents = build_agents(agent_defs, llm)
    user_input = {'code': code_snippet, 'error': error_message}
    tasks = build_tasks(task_defs, agents)
    context = {'user_input': user_input}
    # Run each task, update memory, and pass richer context
    for i, task in enumerate(tasks):
        agent_name = task.agent.role
        # Add agent's memory to context
        context[f"{agent_name}_memory"] = memory[agent_name]
        # Gather inputs for this task
        # Find the corresponding task_def
        task_def = task_defs[i]
        input_keys = task_def['input'] if isinstance(task_def['input'], list) else [task_def['input']]
        input_data = {k: context.get(k, '') for k in input_keys}
        # Run the agent
        result = task.agent.llm.call(f"Task: {task.description}\nInputs: {input_data}\nAgent Memory: {memory[agent_name]}")
        # Save result to context and agent memory
        context[task_def['output']] = result
        memory[agent_name].append({
            'task': task.description,
            'input': input_data,
            'output': result
        })
        print(f"\nDEBUG: Output for {agent_name} ({task.description}):\n{result}\n{'-'*40}")
    save_agent_memory(memory)
    print("\n=== CrewAI Final Output ===\n")
    print(context.get('formatted_explanation', 'No explanation generated.'))
    print("\n=== Educational Tip ===\n")
    print(context.get('educational_tip', 'No tip generated.'))
    print("\n=== Agent Memories (Debug) ===\n")
    for name in agent_names:
        print(f"{name} memory:")
        for entry in memory[name]:
            print(json.dumps(entry, indent=2))
        print("-"*30)

if __name__ == "__main__":
    print("Welcome to the Code Snippet Explainer & Debugger!")
    print("Paste your Python code snippet. End with a blank line:")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    code = "\n".join(lines)
    print("Paste the error message you received. End with a blank line:")
    err_lines = []
    while True:
        line = input()
        if line == "":
            break
        err_lines.append(line)
    error = "\n".join(err_lines)
    run_crewai_workflow(
        agents_yaml="agents.yaml",
        tasks_yaml="tasks.yaml",
        code_snippet=code,
        error_message=error
    )