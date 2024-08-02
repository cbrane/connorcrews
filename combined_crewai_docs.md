

# Agents.md

---
title: crewAI Agents
description: What are crewAI Agents and how to use them.
---

## What is an Agent?
!!! note "What is an Agent?"
    An agent is an **autonomous unit** programmed to:
    <ul>
      <li class='leading-3'>Perform tasks</li>
      <li class='leading-3'>Make decisions</li>
      <li class='leading-3'>Communicate with other agents</li>
    </ul>
      <br/>
    Think of an agent as a member of a team, with specific skills and a particular job to do. Agents can have different roles like 'Researcher', 'Writer', or 'Customer Support', each contributing to the overall goal of the crew.

## Agent Attributes

| Attribute                  | Parameter  | Description                                                                                                                                                                                                                                    |
| :------------------------- | :---- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Role**                   | `role`  | Defines the agent's function within the crew. It determines the kind of tasks the agent is best suited for.                                                                                                                                    |
| **Goal**                   | `goal`  | The individual objective that the agent aims to achieve. It guides the agent's decision-making process.                                                                                                                                        |
| **Backstory**              | `backstory`  | Provides context to the agent's role and goal, enriching the interaction and collaboration dynamics.                                                                                                                                           |
| **LLM** *(optional)*       | `llm`  | Represents the language model that will run the agent. It dynamically fetches the model name from the `OPENAI_MODEL_NAME` environment variable, defaulting to "gpt-4" if not specified.                                                         |
| **Tools** *(optional)*     | `tools`  | Set of capabilities or functions that the agent can use to perform tasks. Expected to be instances of custom classes compatible with the agent's execution environment. Tools are initialized with a default value of an empty list.             |
| **Function Calling LLM** *(optional)* | `function_calling_llm`  | Specifies the language model that will handle the tool calling for this agent, overriding the crew function calling LLM if passed. Default is `None`.                                                                                          |
| **Max Iter** *(optional)*  | `max_iter` | Max Iter is the maximum number of iterations the agent can perform before being forced to give its best answer. Default is `25`.                                                                                                                           |
| **Max RPM** *(optional)*   | `max_rpm`  | Max RPM is the maximum number of requests per minute the agent can perform to avoid rate limits. It's optional and can be left unspecified, with a default value of `None`.                                                                               |
| **Max Execution Time** *(optional)*   | `max_execution_time`  | Max Execution Time is the Maximum execution time for an agent to execute a task. It's optional and can be left unspecified, with a default value of `None`, meaning no max execution time.                                                                     |
| **Verbose** *(optional)*   | `verbose`  | Setting this to `True` configures the internal logger to provide detailed execution logs, aiding in debugging and monitoring. Default is `False`.                                                                                              |
| **Allow Delegation** *(optional)* | `allow_delegation`  | Agents can delegate tasks or questions to one another, ensuring that each task is handled by the most suitable agent. Default is `True`.                                                                                                       |
| **Step Callback** *(optional)* | `step_callback`  | A function that is called after each step of the agent. This can be used to log the agent's actions or to perform other operations. It will overwrite the crew `step_callback`.                                                               |
| **Cache** *(optional)*     | `cache`  | Indicates if the agent should use a cache for tool usage. Default is `True`.                                                                                                                                                                  |
| **System Template** *(optional)*     | `system_template`  | Specifies the system format for the agent. Default is `None`.                                                                                                                                                                  |
| **Prompt Template** *(optional)*     | `prompt_template`  | Specifies the prompt format for the agent. Default is `None`.                                                                                                                                                                  |
| **Response Template** *(optional)*     | `response_template`  | Specifies the response format for the agent. Default is `None`.                                                                                                                                                                  |

## Creating an Agent

!!! note "Agent Interaction"
    Agents can interact with each other using crewAI's built-in delegation and communication mechanisms. This allows for dynamic task management and problem-solving within the crew.

To create an agent, you would typically initialize an instance of the `Agent` class with the desired properties. Here's a conceptual example including all attributes:

```python
# Example: Creating an agent with all attributes
from crewai import Agent

agent = Agent(
  role='Data Analyst',
  goal='Extract actionable insights',
  backstory="""You're a data analyst at a large company.
  You're responsible for analyzing data and providing insights
  to the business.
  You're currently working on a project to analyze the
  performance of our marketing campaigns.""",
  tools=[my_tool1, my_tool2],  # Optional, defaults to an empty list
  llm=my_llm,  # Optional
  function_calling_llm=my_llm,  # Optional
  max_iter=15,  # Optional
  max_rpm=None, # Optional
  max_execution_time=None, # Optional
  verbose=True,  # Optional
  allow_delegation=True,  # Optional
  step_callback=my_intermediate_step_callback,  # Optional
  cache=True,  # Optional
  system_template=my_system_template,  # Optional
  prompt_template=my_prompt_template,  # Optional
  response_template=my_response_template,  # Optional
  config=my_config,  # Optional
  crew=my_crew,  # Optional
  tools_handler=my_tools_handler,  # Optional
  cache_handler=my_cache_handler,  # Optional
  callbacks=[callback1, callback2],  # Optional
  agent_executor=my_agent_executor  # Optional
)
```

## Setting prompt templates

Prompt templates are used to format the prompt for the agent. You can use to update the system, regular and response templates for the agent. Here's an example of how to set prompt templates:

```python
agent = Agent(
        role="{topic} specialist",
        goal="Figure {goal} out",
        backstory="I am the master of {role}",
        system_template="""<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>""",
        prompt_template="""<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>""",
        response_template="""<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>""",
    )
```

## Bring your Third Party Agents
!!! note "Extend your Third Party Agents like LlamaIndex, Langchain, Autogen or fully custom agents using the the crewai's BaseAgent class."

    BaseAgent includes attributes and methods required to integrate with your crews to run and delegate tasks to other agents within your own crew.

    CrewAI is a universal multi agent framework that allows for all agents to work together to automate tasks and solve problems.


```py
from crewai import Agent, Task, Crew
from custom_agent import CustomAgent # You need to build and extend your own agent logic with the CrewAI BaseAgent class then import it here.

from langchain.agents import load_tools

langchain_tools = load_tools(["google-serper"], llm=llm)

agent1 = CustomAgent(
    role="agent role",
    goal="who is {input}?",
    backstory="agent backstory",
    verbose=True,
)

task1 = Task(
    expected_output="a short biography of {input}",
    description="a short biography of {input}",
    agent=agent1,
)

agent2 = Agent(
    role="agent role",
    goal="summarize the short bio for {input} and if needed do more research",
    backstory="agent backstory",
    verbose=True,
)

task2 = Task(
    description="a tldr summary of the short biography",
    expected_output="5 bullet point summary of the biography",
    agent=agent2,
    context=[task1],
)

my_crew = Crew(agents=[agent1, agent2], tasks=[task1, task2])
crew = my_crew.kickoff(inputs={"input": "Mark Twain"})
```


## Conclusion
Agents are the building blocks of the CrewAI framework. By understanding how to define and interact with agents, you can create sophisticated AI systems that leverage the power of collaborative intelligence.


---


# Collaboration.md

---
title: How Agents Collaborate in CrewAI
description: Exploring the dynamics of agent collaboration within the CrewAI framework, focusing on the newly integrated features for enhanced functionality.
---

## Collaboration Fundamentals
!!! note "Core of Agent Interaction"
    Collaboration in CrewAI is fundamental, enabling agents to combine their skills, share information, and assist each other in task execution, embodying a truly cooperative ecosystem.

- **Information Sharing**: Ensures all agents are well-informed and can contribute effectively by sharing data and findings.
- **Task Assistance**: Allows agents to seek help from peers with the required expertise for specific tasks.
- **Resource Allocation**: Optimizes task execution through the efficient distribution and sharing of resources among agents.

## Enhanced Attributes for Improved Collaboration
The `Crew` class has been enriched with several attributes to support advanced functionalities:

- **Language Model Management (`manager_llm`, `function_calling_llm`)**: Manages language models for executing tasks and tools, facilitating sophisticated agent-tool interactions. Note that while `manager_llm` is mandatory for hierarchical processes to ensure proper execution flow, `function_calling_llm` is optional, with a default value provided for streamlined tool interaction.
- **Custom Manager Agent (`manager_agent`)**: Allows specifying a custom agent as the manager instead of using the default manager provided by CrewAI.
- **Process Flow (`process`)**: Defines the execution logic (e.g., sequential, hierarchical) to streamline task distribution and execution.
- **Verbose Logging (`verbose`)**: Offers detailed logging capabilities for monitoring and debugging purposes. It supports both integer and boolean types to indicate the verbosity level. For example, setting `verbose` to 1 might enable basic logging, whereas setting it to True enables more detailed logs.
- **Rate Limiting (`max_rpm`)**: Ensures efficient utilization of resources by limiting requests per minute. Guidelines for setting `max_rpm` should consider the complexity of tasks and the expected load on resources.
- **Internationalization / Customization Support (`language`, `prompt_file`)**: Facilitates full customization of the inner prompts, enhancing global usability. Supported languages and the process for utilizing the `prompt_file` attribute for customization should be clearly documented. [Example of file](https://github.com/joaomdmoura/crewAI/blob/main/src/crewai/translations/en.json)
- **Execution and Output Handling (`full_output`)**: Distinguishes between full and final outputs for nuanced control over task results. Examples showcasing the difference in outputs can aid in understanding the practical implications of this attribute.
- **Callback and Telemetry (`step_callback`, `task_callback`)**: Integrates callbacks for step-wise and task-level execution monitoring, alongside telemetry for performance analytics. The purpose and usage of `task_callback` alongside `step_callback` for granular monitoring should be clearly explained.
- **Crew Sharing (`share_crew`)**: Enables sharing of crew information with CrewAI for continuous improvement and training models. The privacy implications and benefits of this feature, including how it contributes to model improvement, should be outlined.
- **Usage Metrics (`usage_metrics`)**: Stores all metrics for the language model (LLM) usage during all tasks' execution, providing insights into operational efficiency and areas for improvement. Detailed information on accessing and interpreting these metrics for performance analysis should be provided.
- **Memory Usage (`memory`)**: Indicates whether the crew should use memory to store memories of its execution, enhancing task execution and agent learning.
- **Embedder Configuration (`embedder`)**: Specifies the configuration for the embedder to be used by the crew for understanding and generating language. This attribute supports customization of the language model provider.
- **Cache Management (`cache`)**: Determines whether the crew should use a cache to store the results of tool executions, optimizing performance.
- **Output Logging (`output_log_file`)**: Specifies the file path for logging the output of the crew execution.

## Delegation: Dividing to Conquer
Delegation enhances functionality by allowing agents to intelligently assign tasks or seek help, thereby amplifying the crew's overall capability.

## Implementing Collaboration and Delegation
Setting up a crew involves defining the roles and capabilities of each agent. CrewAI seamlessly manages their interactions, ensuring efficient collaboration and delegation, with enhanced customization and monitoring features to adapt to various operational needs.

## Example Scenario
Consider a crew with a researcher agent tasked with data gathering and a writer agent responsible for compiling reports. The integration of advanced language model management and process flow attributes allows for more sophisticated interactions, such as the writer delegating complex research tasks to the researcher or querying specific information, thereby facilitating a seamless workflow.

## Conclusion
The integration of advanced attributes and functionalities into the CrewAI framework significantly enriches the agent collaboration ecosystem. These enhancements not only simplify interactions but also offer unprecedented flexibility and control, paving the way for sophisticated AI-driven solutions capable of tackling complex tasks through intelligent collaboration and delegation.

---


# Crews.md

---
title: crewAI Crews
description: Understanding and utilizing crews in the crewAI framework with comprehensive attributes and functionalities.
---

## What is a Crew?

A crew in crewAI represents a collaborative group of agents working together to achieve a set of tasks. Each crew defines the strategy for task execution, agent collaboration, and the overall workflow.

## Crew Attributes

| Attribute                             | Parameters             | Description                                                                                                                                                                                                                                               |
| :------------------------------------ | :--------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tasks**                             | `tasks`                | A list of tasks assigned to the crew.                                                                                                                                                                                                                     |
| **Agents**                            | `agents`               | A list of agents that are part of the crew.                                                                                                                                                                                                               |
| **Process** _(optional)_              | `process`              | The process flow (e.g., sequential, hierarchical) the crew follows.                                                                                                                                                                                       |
| **Verbose** _(optional)_              | `verbose`              | The verbosity level for logging during execution.                                                                                                                                                                                                         |
| **Manager LLM** _(optional)_          | `manager_llm`          | The language model used by the manager agent in a hierarchical process. **Required when using a hierarchical process.**                                                                                                                                   |
| **Function Calling LLM** _(optional)_ | `function_calling_llm` | If passed, the crew will use this LLM to do function calling for tools for all agents in the crew. Each agent can have its own LLM, which overrides the crew's LLM for function calling.                                                                  |
| **Config** _(optional)_               | `config`               | Optional configuration settings for the crew, in `Json` or `Dict[str, Any]` format.                                                                                                                                                                       |
| **Max RPM** _(optional)_              | `max_rpm`              | Maximum requests per minute the crew adheres to during execution.                                                                                                                                                                                         |
| **Language** _(optional)_             | `language`             | Language used for the crew, defaults to English.                                                                                                                                                                                                          |
| **Language File** _(optional)_        | `language_file`        | Path to the language file to be used for the crew.                                                                                                                                                                                                        |
| **Memory** _(optional)_               | `memory`               | Utilized for storing execution memories (short-term, long-term, entity memory).                                                                                                                                                                           |
| **Cache** _(optional)_                | `cache`                | Specifies whether to use a cache for storing the results of tools' execution.                                                                                                                                                                             |
| **Embedder** _(optional)_             | `embedder`             | Configuration for the embedder to be used by the crew. Mostly used by memory for now.                                                                                                                                                                     |
| **Full Output** _(optional)_          | `full_output`          | Whether the crew should return the full output with all tasks outputs or just the final output.                                                                                                                                                           |
| **Step Callback** _(optional)_        | `step_callback`        | A function that is called after each step of every agent. This can be used to log the agent's actions or to perform other operations; it won't override the agent-specific `step_callback`.                                                               |
| **Task Callback** _(optional)_        | `task_callback`        | A function that is called after the completion of each task. Useful for monitoring or additional operations post-task execution.                                                                                                                          |
| **Share Crew** _(optional)_           | `share_crew`           | Whether you want to share the complete crew information and execution with the crewAI team to make the library better, and allow us to train models.                                                                                                      |
| **Output Log File** _(optional)_      | `output_log_file`      | Whether you want to have a file with the complete crew output and execution. You can set it using True and it will default to the folder you are currently in and it will be called logs.txt or passing a string with the full path and name of the file. |
| **Manager Agent** _(optional)_        | `manager_agent`        | `manager` sets a custom agent that will be used as a manager.                                                                                                                                                                                             |
| **Manager Callbacks** _(optional)_    | `manager_callbacks`    | `manager_callbacks` takes a list of callback handlers to be executed by the manager agent when a hierarchical process is used.                                                                                                                            |
| **Prompt File** _(optional)_          | `prompt_file`          | Path to the prompt JSON file to be used for the crew.                                                                                                                                                                                                     |
| **Planning** *(optional)*             | `planning`             |  Adds planning ability to the Crew. When activated before each Crew iteration, all Crew data is sent to an AgentPlanner that will plan the tasks and this plan will be added to each task description.
| **Planning LLM** *(optional)*         | `planning_llm`         | The language model used by the AgentPlanner in a planning process. |

!!! note "Crew Max RPM"
The `max_rpm` attribute sets the maximum number of requests per minute the crew can perform to avoid rate limits and will override individual agents' `max_rpm` settings if you set it.

## Creating a Crew

When assembling a crew, you combine agents with complementary roles and tools, assign tasks, and select a process that dictates their execution order and interaction.

### Example: Assembling a Crew

```python
from crewai import Crew, Agent, Task, Process
from langchain_community.tools import DuckDuckGoSearchRun
from crewai_tools import tool

@tool('DuckDuckGoSearch')
def search(search_query: str):
    """Search the web for information on a given topic"""
    return DuckDuckGoSearchRun().run(search_query)

# Define agents with specific roles and tools
researcher = Agent(
    role='Senior Research Analyst',
    goal='Discover innovative AI technologies',
    backstory="""You're a senior research analyst at a large company.
        You're responsible for analyzing data and providing insights
        to the business.
        You're currently working on a project to analyze the
        trends and innovations in the space of artificial intelligence.""",
    tools=[search]
)

writer = Agent(
    role='Content Writer',
    goal='Write engaging articles on AI discoveries',
    backstory="""You're a senior writer at a large company.
        You're responsible for creating content to the business.
        You're currently working on a project to write about trends
        and innovations in the space of AI for your next meeting.""",
    verbose=True
)

# Create tasks for the agents
research_task = Task(
    description='Identify breakthrough AI technologies',
    agent=researcher,
    expected_output='A bullet list summary of the top 5 most important AI news'
)
write_article_task = Task(
    description='Draft an article on the latest AI technologies',
    agent=writer,
    expected_output='3 paragraph blog post on the latest AI technologies'
)

# Assemble the crew with a sequential process
my_crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_article_task],
    process=Process.sequential,
    full_output=True,
    verbose=True,
)
```

## Crew Output

!!! note "Understanding Crew Outputs"
The output of a crew in the crewAI framework is encapsulated within the `CrewOutput` class.
This class provides a structured way to access results of the crew's execution, including various formats such as raw strings, JSON, and Pydantic models.
The `CrewOutput` includes the results from the final task output, token usage, and individual task outputs.

### Crew Output Attributes

| Attribute        | Parameters     | Type                       | Description                                                                                          |
| :--------------- | :------------- | :------------------------- | :--------------------------------------------------------------------------------------------------- |
| **Raw**          | `raw`          | `str`                      | The raw output of the crew. This is the default format for the output.                               |
| **Pydantic**     | `pydantic`     | `Optional[BaseModel]`      | A Pydantic model object representing the structured output of the crew.                              |
| **JSON Dict**    | `json_dict`    | `Optional[Dict[str, Any]]` | A dictionary representing the JSON output of the crew.                                               |
| **Tasks Output** | `tasks_output` | `List[TaskOutput]`         | A list of `TaskOutput` objects, each representing the output of a task in the crew.                  |
| **Token Usage**  | `token_usage`  | `Dict[str, Any]`           | A summary of token usage, providing insights into the language model's performance during execution. |

### Crew Output Methods and Properties

| Method/Property | Description                                                                                       |
| :-------------- | :------------------------------------------------------------------------------------------------ |
| **json**        | Returns the JSON string representation of the crew output if the output format is JSON.           |
| **to_dict**     | Converts the JSON and Pydantic outputs to a dictionary.                                           |
| \***\*str\*\*** | Returns the string representation of the crew output, prioritizing Pydantic, then JSON, then raw. |

### Accessing Crew Outputs

Once a crew has been executed, its output can be accessed through the `output` attribute of the `Crew` object. The `CrewOutput` class provides various ways to interact with and present this output.

#### Example

```python
# Example crew execution
crew = Crew(
    agents=[research_agent, writer_agent],
    tasks=[research_task, write_article_task],
    verbose=2
)

crew_output = crew.kickoff()

# Accessing the crew output
print(f"Raw Output: {crew_output.raw}")
if crew_output.json_dict:
    print(f"JSON Output: {json.dumps(crew_output.json_dict, indent=2)}")
if crew_output.pydantic:
    print(f"Pydantic Output: {crew_output.pydantic}")
print(f"Tasks Output: {crew_output.tasks_output}")
print(f"Token Usage: {crew_output.token_usage}")
```

## Memory Utilization

Crews can utilize memory (short-term, long-term, and entity memory) to enhance their execution and learning over time. This feature allows crews to store and recall execution memories, aiding in decision-making and task execution strategies.

## Cache Utilization

Caches can be employed to store the results of tools' execution, making the process more efficient by reducing the need to re-execute identical tasks.

## Crew Usage Metrics

After the crew execution, you can access the `usage_metrics` attribute to view the language model (LLM) usage metrics for all tasks executed by the crew. This provides insights into operational efficiency and areas for improvement.

```python
# Access the crew's usage metrics
crew = Crew(agents=[agent1, agent2], tasks=[task1, task2])
crew.kickoff()
print(crew.usage_metrics)
```

## Crew Execution Process

- **Sequential Process**: Tasks are executed one after another, allowing for a linear flow of work.
- **Hierarchical Process**: A manager agent coordinates the crew, delegating tasks and validating outcomes before proceeding. **Note**: A `manager_llm` or `manager_agent` is required for this process and it's essential for validating the process flow.

### Kicking Off a Crew

Once your crew is assembled, initiate the workflow with the `kickoff()` method. This starts the execution process according to the defined process flow.

```python
# Start the crew's task execution
result = my_crew.kickoff()
print(result)
```

### Different ways to Kicking Off a Crew

Once your crew is assembled, initiate the workflow with the appropriate kickoff method. CrewAI provides several methods for better control over the kickoff process: `kickoff()`, `kickoff_for_each()`, `kickoff_async()`, and `kickoff_for_each_async()`.

`kickoff()`: Starts the execution process according to the defined process flow.
`kickoff_for_each()`: Executes tasks for each agent individually.
`kickoff_async()`: Initiates the workflow asynchronously.
`kickoff_for_each_async()`: Executes tasks for each agent individually in an asynchronous manner.

```python
# Start the crew's task execution
result = my_crew.kickoff()
print(result)

# Example of using kickoff_for_each
inputs_array = [{'topic': 'AI in healthcare'}, {'topic': 'AI in finance'}]
results = my_crew.kickoff_for_each(inputs=inputs_array)
for result in results:
    print(result)

# Example of using kickoff_async
inputs = {'topic': 'AI in healthcare'}
async_result = my_crew.kickoff_async(inputs=inputs)
print(async_result)

# Example of using kickoff_for_each_async
inputs_array = [{'topic': 'AI in healthcare'}, {'topic': 'AI in finance'}]
async_results = my_crew.kickoff_for_each_async(inputs=inputs_array)
for async_result in async_results:
    print(async_result)
```

These methods provide flexibility in how you manage and execute tasks within your crew, allowing for both synchronous and asynchronous workflows tailored to your needs


### Replaying from specific task:
You can now replay from a specific task using our cli command replay.

The replay feature in CrewAI allows you to replay from a specific task using the command-line interface (CLI). By running the command `crewai replay -t <task_id>`, you can specify the `task_id` for the replay process.

Kickoffs will now save the latest kickoffs returned task outputs locally for you to be able to replay from.


### Replaying from specific task Using the CLI
To use the replay feature, follow these steps:

1. Open your terminal or command prompt.
2. Navigate to the directory where your CrewAI project is located.
3. Run the following command:

To view latest kickoff task_ids use:

```shell
crewai log-tasks-outputs
```


```shell
crewai replay -t <task_id>
```

These commands let you replay from your latest kickoff tasks, still retaining context from previously executed tasks.


---


# Memory.md

---
title: crewAI Memory Systems
description: Leveraging memory systems in the crewAI framework to enhance agent capabilities.
---

## Introduction to Memory Systems in crewAI
!!! note "Enhancing Agent Intelligence"
    The crewAI framework introduces a sophisticated memory system designed to significantly enhance the capabilities of AI agents. This system comprises short-term memory, long-term memory, entity memory, and contextual memory, each serving a unique purpose in aiding agents to remember, reason, and learn from past interactions.

## Memory System Components

| Component            | Description                                                  |
| :------------------- | :----------------------------------------------------------- |
| **Short-Term Memory**| Temporarily stores recent interactions and outcomes, enabling agents to recall and utilize information relevant to their current context during the current executions. |
| **Long-Term Memory** | Preserves valuable insights and learnings from past executions, allowing agents to build and refine their knowledge over time. So Agents can remember what they did right and wrong across multiple executions |
| **Entity Memory**    | Captures and organizes information about entities (people, places, concepts) encountered during tasks, facilitating deeper understanding and relationship mapping. |
| **Contextual Memory**| Maintains the context of interactions by combining `ShortTermMemory`, `LongTermMemory`, and `EntityMemory`, aiding in the coherence and relevance of agent responses over a sequence of tasks or a conversation. |

## How Memory Systems Empower Agents

1. **Contextual Awareness**: With short-term and contextual memory, agents gain the ability to maintain context over a conversation or task sequence, leading to more coherent and relevant responses.

2. **Experience Accumulation**: Long-term memory allows agents to accumulate experiences, learning from past actions to improve future decision-making and problem-solving.

3. **Entity Understanding**: By maintaining entity memory, agents can recognize and remember key entities, enhancing their ability to process and interact with complex information.

## Implementing Memory in Your Crew

When configuring a crew, you can enable and customize each memory component to suit the crew's objectives and the nature of tasks it will perform.
By default, the memory system is disabled, and you can ensure it is active by setting `memory=True` in the crew configuration. The memory will use OpenAI Embeddings by default, but you can change it by setting `embedder` to a different model.

The 'embedder' only applies to **Short-Term Memory** which uses Chroma for RAG using EmbedChain package.  
The **Long-Term Memory** uses SQLLite3 to store task results.  Currently, there is no way to override these storage implementations.
The data storage files are saved into a platform specific location found using the appdirs package 
and the name of the project which can be overridden using the **CREWAI_STORAGE_DIR** environment variable.

### Example: Configuring Memory for a Crew

```python
from crewai import Crew, Agent, Task, Process

# Assemble your crew with memory capabilities
my_crew = Crew(
    agents=[...],
    tasks=[...],
    process=Process.sequential,
    memory=True,
    verbose=True
)
```

## Additional Embedding Providers

### Using OpenAI embeddings (already default)
```python
from crewai import Crew, Agent, Task, Process

my_crew = Crew(
		agents=[...],
		tasks=[...],
		process=Process.sequential,
		memory=True,
		verbose=True,
		embedder={
				"provider": "openai",
				"config":{
						"model": 'text-embedding-3-small'
				}
		}
)
```

### Using Google AI embeddings
```python
from crewai import Crew, Agent, Task, Process

my_crew = Crew(
		agents=[...],
		tasks=[...],
		process=Process.sequential,
		memory=True,
		verbose=True,
		embedder={
			"provider": "google",
			"config":{
				"model": 'models/embedding-001',
				"task_type": "retrieval_document",
				"title": "Embeddings for Embedchain"
			}
		}
)
```

### Using Azure OpenAI embeddings
```python
from crewai import Crew, Agent, Task, Process

my_crew = Crew(
		agents=[...],
		tasks=[...],
		process=Process.sequential,
		memory=True,
		verbose=True,
		embedder={
			"provider": "azure_openai",
			"config":{
				"model": 'text-embedding-ada-002',
				"deployment_name": "you_embedding_model_deployment_name"
			}
		}
)
```

### Using GPT4ALL embeddings
```python
from crewai import Crew, Agent, Task, Process

my_crew = Crew(
		agents=[...],
		tasks=[...],
		process=Process.sequential,
		memory=True,
		verbose=True,
		embedder={
			"provider": "gpt4all"
		}
)
```

### Using Vertex AI embeddings
```python
from crewai import Crew, Agent, Task, Process

my_crew = Crew(
		agents=[...],
		tasks=[...],
		process=Process.sequential,
		memory=True,
		verbose=True,
		embedder={
			"provider": "vertexai",
			"config":{
				"model": 'textembedding-gecko'
			}
		}
)
```

### Using Cohere embeddings
```python
from crewai import Crew, Agent, Task, Process

my_crew = Crew(
		agents=[...],
		tasks=[...],
		process=Process.sequential,
		memory=True,
		verbose=True,
		embedder={
			"provider": "cohere",
			"config":{
				"model": "embed-english-v3.0"
    		"vector_dimension": 1024
			}
		}
)
```

### Resetting Memory
```sh
crewai reset_memories [OPTIONS]
```

#### Resetting Memory Options
- **`-l, --long`**
  - **Description:** Reset LONG TERM memory.
  - **Type:** Flag (boolean)
  - **Default:** False

- **`-s, --short`**
  - **Description:** Reset SHORT TERM memory.
  - **Type:** Flag (boolean)
  - **Default:** False

- **`-e, --entities`**
  - **Description:** Reset ENTITIES memory.
  - **Type:** Flag (boolean)
  - **Default:** False

- **`-k, --kickoff-outputs`**
  - **Description:** Reset LATEST KICKOFF TASK OUTPUTS.
  - **Type:** Flag (boolean)
  - **Default:** False

- **`-a, --all`**
  - **Description:** Reset ALL memories.
  - **Type:** Flag (boolean)
  - **Default:** False



## Benefits of Using crewAI's Memory System
- **Adaptive Learning:** Crews become more efficient over time, adapting to new information and refining their approach to tasks.
- **Enhanced Personalization:** Memory enables agents to remember user preferences and historical interactions, leading to personalized experiences.
- **Improved Problem Solving:** Access to a rich memory store aids agents in making more informed decisions, drawing on past learnings and contextual insights.

## Getting Started
Integrating crewAI's memory system into your projects is straightforward. By leveraging the provided memory components and configurations, you can quickly empower your agents with the ability to remember, reason, and learn from their interactions, unlocking new levels of intelligence and capability.


---


# Planning.md

---
title: crewAI Planning
description: Learn how to add planning to your crewAI Crew and improve their performance.
---

## Introduction
The planning feature in CrewAI allows you to add planning capability to your crew. When enabled, before each Crew iteration, all Crew information is sent to an AgentPlanner that will plan the tasks step by step, and this plan will be added to each task description.

### Using the Planning Feature
Getting started with the planning feature is very easy, the only step required is to add `planning=True` to your Crew:

```python
from crewai import Crew, Agent, Task, Process

# Assemble your crew with planning capabilities
my_crew = Crew(
    agents=self.agents,
    tasks=self.tasks,
    process=Process.sequential,
    planning=True,
)
```

From this point on, your crew will have planning enabled, and the tasks will be planned before each iteration.

#### Planning LLM

Now you can define the LLM that will be used to plan the tasks. You can use any ChatOpenAI LLM model available.

```python
from crewai import Crew, Agent, Task, Process
from langchain_openai import ChatOpenAI

# Assemble your crew with planning capabilities and custom LLM
my_crew = Crew(
    agents=self.agents,
    tasks=self.tasks,
    process=Process.sequential,
    planning=True,
    planning_llm=ChatOpenAI(model="gpt-4o")
)
```


### Example

When running the base case example, you will see something like the following output, which represents the output of the AgentPlanner responsible for creating the step-by-step logic to add to the Agents tasks.

```bash

[2024-07-15 16:49:11][INFO]: Planning the crew execution
**Step-by-Step Plan for Task Execution**

**Task Number 1: Conduct a thorough research about AI LLMs**

**Agent:** AI LLMs Senior Data Researcher

**Agent Goal:** Uncover cutting-edge developments in AI LLMs

**Task Expected Output:** A list with 10 bullet points of the most relevant information about AI LLMs

**Task Tools:** None specified

**Agent Tools:** None specified

**Step-by-Step Plan:**

1. **Define Research Scope:**
   - Determine the specific areas of AI LLMs to focus on, such as advancements in architecture, use cases, ethical considerations, and performance metrics.

2. **Identify Reliable Sources:**
   - List reputable sources for AI research, including academic journals, industry reports, conferences (e.g., NeurIPS, ACL), AI research labs (e.g., OpenAI, Google AI), and online databases (e.g., IEEE Xplore, arXiv).

3. **Collect Data:**
   - Search for the latest papers, articles, and reports published in 2023 and early 2024.
   - Use keywords like "Large Language Models 2024", "AI LLM advancements", "AI ethics 2024", etc.

4. **Analyze Findings:**
   - Read and summarize the key points from each source.
   - Highlight new techniques, models, and applications introduced in the past year.

5. **Organize Information:**
   - Categorize the information into relevant topics (e.g., new architectures, ethical implications, real-world applications).
   - Ensure each bullet point is concise but informative.

6. **Create the List:**
   - Compile the 10 most relevant pieces of information into a bullet point list.
   - Review the list to ensure clarity and relevance.

**Expected Output:**
A list with 10 bullet points of the most relevant information about AI LLMs.

---

**Task Number 2: Review the context you got and expand each topic into a full section for a report**

**Agent:** AI LLMs Reporting Analyst

**Agent Goal:** Create detailed reports based on AI LLMs data analysis and research findings

**Task Expected Output:** A fully fledge report with the main topics, each with a full section of information. Formatted as markdown without '```'

**Task Tools:** None specified

**Agent Tools:** None specified

**Step-by-Step Plan:**

1. **Review the Bullet Points:**
   - Carefully read through the list of 10 bullet points provided by the AI LLMs Senior Data Researcher.

2. **Outline the Report:**
   - Create an outline with each bullet point as a main section heading.
   - Plan sub-sections under each main heading to cover different aspects of the topic.

3. **Research Further Details:**
   - For each bullet point, conduct additional research if necessary to gather more detailed information.
   - Look for case studies, examples, and statistical data to support each section.

4. **Write Detailed Sections:**
   - Expand each bullet point into a comprehensive section.
   - Ensure each section includes an introduction, detailed explanation, examples, and a conclusion.
   - Use markdown formatting for headings, subheadings, lists, and emphasis.

5. **Review and Edit:**
   - Proofread the report for clarity, coherence, and correctness.
   - Make sure the report flows logically from one section to the next.
   - Format the report according to markdown standards.

6. **Finalize the Report:**
   - Ensure the report is complete with all sections expanded and detailed.
   - Double-check formatting and make any necessary adjustments.

**Expected Output:**
A fully-fledged report with the main topics, each with a full section of information. Formatted as markdown without '```'.

---
```


---


# Processes.md

---
title: Managing Processes in CrewAI
description: Detailed guide on workflow management through processes in CrewAI, with updated implementation details.
---

## Understanding Processes
!!! note "Core Concept"
    In CrewAI, processes orchestrate the execution of tasks by agents, akin to project management in human teams. These processes ensure tasks are distributed and executed efficiently, in alignment with a predefined strategy.

## Process Implementations

- **Sequential**: Executes tasks sequentially, ensuring tasks are completed in an orderly progression.
- **Hierarchical**: Organizes tasks in a managerial hierarchy, where tasks are delegated and executed based on a structured chain of command. A manager language model (`manager_llm`) or a custom manager agent (`manager_agent`) must be specified in the crew to enable the hierarchical process, facilitating the creation and management of tasks by the manager.
- **Consensual Process (Planned)**: Aiming for collaborative decision-making among agents on task execution, this process type introduces a democratic approach to task management within CrewAI. It is planned for future development and is not currently implemented in the codebase.

## The Role of Processes in Teamwork
Processes enable individual agents to operate as a cohesive unit, streamlining their efforts to achieve common objectives with efficiency and coherence.

## Assigning Processes to a Crew
To assign a process to a crew, specify the process type upon crew creation to set the execution strategy. For a hierarchical process, ensure to define `manager_llm` or `manager_agent` for the manager agent.

```python
from crewai import Crew
from crewai.process import Process
from langchain_openai import ChatOpenAI

# Example: Creating a crew with a sequential process
crew = Crew(
    agents=my_agents,
    tasks=my_tasks,
    process=Process.sequential
)

# Example: Creating a crew with a hierarchical process
# Ensure to provide a manager_llm or manager_agent
crew = Crew(
    agents=my_agents,
    tasks=my_tasks,
    process=Process.hierarchical,
    manager_llm=ChatOpenAI(model="gpt-4")
    # or
    # manager_agent=my_manager_agent
)
```
**Note:** Ensure `my_agents` and `my_tasks` are defined prior to creating a `Crew` object, and for the hierarchical process, either `manager_llm` or `manager_agent` is also required.

## Sequential Process
This method mirrors dynamic team workflows, progressing through tasks in a thoughtful and systematic manner. Task execution follows the predefined order in the task list, with the output of one task serving as context for the next.

To customize task context, utilize the `context` parameter in the `Task` class to specify outputs that should be used as context for subsequent tasks.

## Hierarchical Process
Emulates a corporate hierarchy, CrewAI allows specifying a custom manager agent or automatically creates one, requiring the specification of a manager language model (`manager_llm`). This agent oversees task execution, including planning, delegation, and validation. Tasks are not pre-assigned; the manager allocates tasks to agents based on their capabilities, reviews outputs, and assesses task completion.

## Process Class: Detailed Overview
The `Process` class is implemented as an enumeration (`Enum`), ensuring type safety and restricting process values to the defined types (`sequential`, `hierarchical`). The consensual process is planned for future inclusion, emphasizing our commitment to continuous development and innovation.

## Additional Task Features
- **Asynchronous Execution**: Tasks can now be executed asynchronously, allowing for parallel processing and efficiency improvements. This feature is designed to enable tasks to be carried out concurrently, enhancing the overall productivity of the crew.
- **Human Input Review**: An optional feature that enables the review of task outputs by humans to ensure quality and accuracy before finalization. This additional step introduces a layer of oversight, providing an opportunity for human intervention and validation.
- **Output Customization**: Tasks support various output formats, including JSON (`output_json`), Pydantic models (`output_pydantic`), and file outputs (`output_file`), providing flexibility in how task results are captured and utilized. This allows for a wide range of output possibilities, catering to different needs and requirements.

## Conclusion
The structured collaboration facilitated by processes within CrewAI is crucial for enabling systematic teamwork among agents. This documentation has been updated to reflect the latest features, enhancements, and the planned integration of the Consensual Process, ensuring users have access to the most current and comprehensive information.

---


# Tasks.md

---
title: crewAI Tasks
description: Detailed guide on managing and creating tasks within the crewAI framework, reflecting the latest codebase updates.
---

## Overview of a Task

!!! note "What is a Task?"
In the crewAI framework, tasks are specific assignments completed by agents. They provide all necessary details for execution, such as a description, the agent responsible, required tools, and more, facilitating a wide range of action complexities.

Tasks within crewAI can be collaborative, requiring multiple agents to work together. This is managed through the task properties and orchestrated by the Crew's process, enhancing teamwork and efficiency.

## Task Attributes

| Attribute                        | Parameters        | Description                                                                                                          |
| :------------------------------- | :---------------- | :------------------------------------------------------------------------------------------------------------------- |
| **Description**                  | `description`     | A clear, concise statement of what the task entails.                                                                 |
| **Agent**                        | `agent`           | The agent responsible for the task, assigned either directly or by the crew's process.                               |
| **Expected Output**              | `expected_output` | A detailed description of what the task's completion looks like.                                                     |
| **Tools** _(optional)_           | `tools`           | The functions or capabilities the agent can utilize to perform the task.                                             |
| **Async Execution** _(optional)_ | `async_execution` | If set, the task executes asynchronously, allowing progression without waiting for completion.                       |
| **Context** _(optional)_         | `context`         | Specifies tasks whose outputs are used as context for this task.                                                     |
| **Config** _(optional)_          | `config`          | Additional configuration details for the agent executing the task, allowing further customization.                   |
| **Output JSON** _(optional)_     | `output_json`     | Outputs a JSON object, requiring an OpenAI client. Only one output format can be set.                                |
| **Output Pydantic** _(optional)_ | `output_pydantic` | Outputs a Pydantic model object, requiring an OpenAI client. Only one output format can be set.                      |
| **Output File** _(optional)_     | `output_file`     | Saves the task output to a file. If used with `Output JSON` or `Output Pydantic`, specifies how the output is saved. |
| **Output** _(optional)_          | `output`          | The output of the task, containing the raw, JSON, and Pydantic output plus additional details.                       |
| **Callback** _(optional)_        | `callback`        | A Python callable that is executed with the task's output upon completion.                                           |
| **Human Input** _(optional)_     | `human_input`     | Indicates if the task requires human feedback at the end, useful for tasks needing human oversight.                  |

## Creating a Task

Creating a task involves defining its scope, responsible agent, and any additional attributes for flexibility:

```python
from crewai import Task

task = Task(
    description='Find and summarize the latest and most relevant news on AI',
    agent=sales_agent,
    expected_output='A bullet list summary of the top 5 most important AI news',
)
```

!!! note "Task Assignment"
Directly specify an `agent` for assignment or let the `hierarchical` CrewAI's process decide based on roles, availability, etc.

## Task Output

!!! note "Understanding Task Outputs"
The output of a task in the crewAI framework is encapsulated within the `TaskOutput` class. This class provides a structured way to access results of a task, including various formats such as raw strings, JSON, and Pydantic models.
By default, the `TaskOutput` will only include the `raw` output. A `TaskOutput` will only include the `pydantic` or `json_dict` output if the original `Task` object was configured with `output_pydantic` or `output_json`, respectively.

### Task Output Attributes

| Attribute         | Parameters      | Type                       | Description                                                                                        |
| :---------------- | :-------------- | :------------------------- | :------------------------------------------------------------------------------------------------- |
| **Description**   | `description`   | `str`                      | A brief description of the task.                                                                   |
| **Summary**       | `summary`       | `Optional[str]`            | A short summary of the task, auto-generated from the description.                                  |
| **Raw**           | `raw`           | `str`                      | The raw output of the task. This is the default format for the output.                             |
| **Pydantic**      | `pydantic`      | `Optional[BaseModel]`      | A Pydantic model object representing the structured output of the task.                            |
| **JSON Dict**     | `json_dict`     | `Optional[Dict[str, Any]]` | A dictionary representing the JSON output of the task.                                             |
| **Agent**         | `agent`         | `str`                      | The agent that executed the task.                                                                  |
| **Output Format** | `output_format` | `OutputFormat`             | The format of the task output, with options including RAW, JSON, and Pydantic. The default is RAW. |

### Task Output Methods and Properties

| Method/Property | Description                                                                                       |
| :-------------- | :------------------------------------------------------------------------------------------------ |
| **json**        | Returns the JSON string representation of the task output if the output format is JSON.           |
| **to_dict**     | Converts the JSON and Pydantic outputs to a dictionary.                                           |
| \***\*str\*\*** | Returns the string representation of the task output, prioritizing Pydantic, then JSON, then raw. |

### Accessing Task Outputs

Once a task has been executed, its output can be accessed through the `output` attribute of the `Task` object. The `TaskOutput` class provides various ways to interact with and present this output.

#### Example

```python
# Example task
task = Task(
    description='Find and summarize the latest AI news',
    expected_output='A bullet list summary of the top 5 most important AI news',
    agent=research_agent,
    tools=[search_tool]
)

# Execute the crew
crew = Crew(
    agents=[research_agent],
    tasks=[task],
    verbose=2
)

result = crew.kickoff()

# Accessing the task output
task_output = task.output

print(f"Task Description: {task_output.description}")
print(f"Task Summary: {task_output.summary}")
print(f"Raw Output: {task_output.raw}")
if task_output.json_dict:
    print(f"JSON Output: {json.dumps(task_output.json_dict, indent=2)}")
if task_output.pydantic:
    print(f"Pydantic Output: {task_output.pydantic}")
```

## Integrating Tools with Tasks

Leverage tools from the [crewAI Toolkit](https://github.com/joaomdmoura/crewai-tools) and [LangChain Tools](https://python.langchain.com/docs/integrations/tools) for enhanced task performance and agent interaction.

## Creating a Task with Tools

```python
import os
os.environ["OPENAI_API_KEY"] = "Your Key"
os.environ["SERPER_API_KEY"] = "Your Key" # serper.dev API key

from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool

research_agent = Agent(
  role='Researcher',
  goal='Find and summarize the latest AI news',
  backstory="""You're a researcher at a large company.
  You're responsible for analyzing data and providing insights
  to the business.""",
  verbose=True
)

search_tool = SerperDevTool()

task = Task(
  description='Find and summarize the latest AI news',
  expected_output='A bullet list summary of the top 5 most important AI news',
  agent=research_agent,
  tools=[search_tool]
)

crew = Crew(
    agents=[research_agent],
    tasks=[task],
    verbose=2
)

result = crew.kickoff()
print(result)
```

This demonstrates how tasks with specific tools can override an agent's default set for tailored task execution.

## Referring to Other Tasks

In crewAI, the output of one task is automatically relayed into the next one, but you can specifically define what tasks' output, including multiple, should be used as context for another task.

This is useful when you have a task that depends on the output of another task that is not performed immediately after it. This is done through the `context` attribute of the task:

```python
# ...

research_ai_task = Task(
    description='Find and summarize the latest AI news',
    expected_output='A bullet list summary of the top 5 most important AI news',
    async_execution=True,
    agent=research_agent,
    tools=[search_tool]
)

research_ops_task = Task(
    description='Find and summarize the latest AI Ops news',
    expected_output='A bullet list summary of the top 5 most important AI Ops news',
    async_execution=True,
    agent=research_agent,
    tools=[search_tool]
)

write_blog_task = Task(
    description="Write a full blog post about the importance of AI and its latest news",
    expected_output='Full blog post that is 4 paragraphs long',
    agent=writer_agent,
    context=[research_ai_task, research_ops_task]
)

#...
```

## Asynchronous Execution

You can define a task to be executed asynchronously. This means that the crew will not wait for it to be completed to continue with the next task. This is useful for tasks that take a long time to be completed, or that are not crucial for the next tasks to be performed.

You can then use the `context` attribute to define in a future task that it should wait for the output of the asynchronous task to be completed.

```python
#...

list_ideas = Task(
    description="List of 5 interesting ideas to explore for an article about AI.",
    expected_output="Bullet point list of 5 ideas for an article.",
    agent=researcher,
    async_execution=True # Will be executed asynchronously
)

list_important_history = Task(
    description="Research the history of AI and give me the 5 most important events.",
    expected_output="Bullet point list of 5 important events.",
    agent=researcher,
    async_execution=True # Will be executed asynchronously
)

write_article = Task(
    description="Write an article about AI, its history, and interesting ideas.",
    expected_output="A 4 paragraph article about AI.",
    agent=writer,
    context=[list_ideas, list_important_history] # Will wait for the output of the two tasks to be completed
)

#...
```

## Callback Mechanism

The callback function is executed after the task is completed, allowing for actions or notifications to be triggered based on the task's outcome.

```python
# ...

def callback_function(output: TaskOutput):
    # Do something after the task is completed
    # Example: Send an email to the manager
    print(f"""
        Task completed!
        Task: {output.description}
        Output: {output.raw_output}
    """)

research_task = Task(
    description='Find and summarize the latest AI news',
    expected_output='A bullet list summary of the top 5 most important AI news',
    agent=research_agent,
    tools=[search_tool],
    callback=callback_function
)

#...
```

## Accessing a Specific Task Output

Once a crew finishes running, you can access the output of a specific task by using the `output` attribute of the task object:

```python
# ...
task1 = Task(
    description='Find and summarize the latest AI news',
    expected_output='A bullet list summary of the top 5 most important AI news',
    agent=research_agent,
    tools=[search_tool]
)

#...

crew = Crew(
    agents=[research_agent],
    tasks=[task1, task2, task3],
    verbose=2
)

result = crew.kickoff()

# Returns a TaskOutput object with the description and results of the task
print(f"""
    Task completed!
    Task: {task1.output.description}
    Output: {task1.output.raw_output}
""")
```

## Tool Override Mechanism

Specifying tools in a task allows for dynamic adaptation of agent capabilities, emphasizing CrewAI's flexibility.

## Error Handling and Validation Mechanisms

While creating and executing tasks, certain validation mechanisms are in place to ensure the robustness and reliability of task attributes. These include but are not limited to:

- Ensuring only one output type is set per task to maintain clear output expectations.
- Preventing the manual assignment of the `id` attribute to uphold the integrity of the unique identifier system.

These validations help in maintaining the consistency and reliability of task executions within the crewAI framework.

## Creating Directories when Saving Files

You can now specify if a task should create directories when saving its output to a file. This is particularly useful for organizing outputs and ensuring that file paths are correctly structured.

```python
# ...

save_output_task = Task(
    description='Save the summarized AI news to a file',
    expected_output='File saved successfully',
    agent=research_agent,
    tools=[file_save_tool],
    output_file='outputs/ai_news_summary.txt',
    create_directory=True
)

#...
```

## Conclusion

Tasks are the driving force behind the actions of agents in crewAI. By properly defining tasks and their outcomes, you set the stage for your AI agents to work effectively, either independently or as a collaborative unit. Equipping tasks with appropriate tools, understanding the execution process, and following robust validation practices are crucial for maximizing CrewAI's potential, ensuring agents are effectively prepared for their assignments and that tasks are executed as intended.


---


# Testing.md

---
title: crewAI Testing
description: Learn how to test your crewAI Crew and evaluate their performance.
---

## Introduction

Testing is a crucial part of the development process, and it is essential to ensure that your crew is performing as expected. And with crewAI, you can easily test your crew and evaluate its performance using the built-in testing capabilities.

### Using the Testing Feature

We added the CLI command `crewai test` to make it easy to test your crew. This command will run your crew for a specified number of iterations and provide detailed performance metrics.
The parameters are `n_iterations` and `model` which are optional and default to 2 and `gpt-4o-mini` respectively. For now the only provider available is OpenAI.

```bash
crewai test
```

If you want to run more iterations or use a different model, you can specify the parameters like this:

```bash
crewai test --n_iterations 5 --model gpt-4o
```

What happens when you run the `crewai test` command is that the crew will be executed for the specified number of iterations, and the performance metrics will be displayed at the end of the run.

A table of scores at the end will show the performance of the crew in terms of the following metrics:
```
                Task Scores
          (1-10 Higher is better)
┏━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ Tasks/Crew ┃ Run 1 ┃ Run 2 ┃ Avg. Total ┃
┡━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│ Task 1     │ 10.0  │ 9.0   │ 9.5        │
│ Task 2     │ 9.0   │ 9.0   │ 9.0        │
│ Crew       │ 9.5   │ 9.0   │ 9.2        │
└────────────┴───────┴───────┴────────────┘
```

The example above shows the test results for two runs of the crew with two tasks, with the average total score for each task and the crew as a whole.



---


# Tools.md

---
title: crewAI Tools
description: Understanding and leveraging tools within the crewAI framework for agent collaboration and task execution.
---

## Introduction
CrewAI tools empower agents with capabilities ranging from web searching and data analysis to collaboration and delegating tasks among coworkers. This documentation outlines how to create, integrate, and leverage these tools within the CrewAI framework, including a new focus on collaboration tools.

## What is a Tool?
!!! note "Definition"
    A tool in CrewAI is a skill or function that agents can utilize to perform various actions. This includes tools from the [crewAI Toolkit](https://github.com/joaomdmoura/crewai-tools) and [LangChain Tools](https://python.langchain.com/docs/integrations/tools), enabling everything from simple searches to complex interactions and effective teamwork among agents.

## Key Characteristics of Tools

- **Utility**: Crafted for tasks such as web searching, data analysis, content generation, and agent collaboration.
- **Integration**: Boosts agent capabilities by seamlessly integrating tools into their workflow.
- **Customizability**: Provides the flexibility to develop custom tools or utilize existing ones, catering to the specific needs of agents.
- **Error Handling**: Incorporates robust error handling mechanisms to ensure smooth operation.
- **Caching Mechanism**: Features intelligent caching to optimize performance and reduce redundant operations.

## Using crewAI Tools

To enhance your agents' capabilities with crewAI tools, begin by installing our extra tools package:

```bash
pip install 'crewai[tools]'
```

Here's an example demonstrating their use:

```python
import os
from crewai import Agent, Task, Crew
# Importing crewAI tools
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool
)

# Set up API keys
os.environ["SERPER_API_KEY"] = "Your Key" # serper.dev API key
os.environ["OPENAI_API_KEY"] = "Your Key"

# Instantiate tools
docs_tool = DirectoryReadTool(directory='./blog-posts')
file_tool = FileReadTool()
search_tool = SerperDevTool()
web_rag_tool = WebsiteSearchTool()

# Create agents
researcher = Agent(
    role='Market Research Analyst',
    goal='Provide up-to-date market analysis of the AI industry',
    backstory='An expert analyst with a keen eye for market trends.',
    tools=[search_tool, web_rag_tool],
    verbose=True
)

writer = Agent(
    role='Content Writer',
    goal='Craft engaging blog posts about the AI industry',
    backstory='A skilled writer with a passion for technology.',
    tools=[docs_tool, file_tool],
    verbose=True
)

# Define tasks
research = Task(
    description='Research the latest trends in the AI industry and provide a summary.',
    expected_output='A summary of the top 3 trending developments in the AI industry with a unique perspective on their significance.',
    agent=researcher
)

write = Task(
    description='Write an engaging blog post about the AI industry, based on the research analyst’s summary. Draw inspiration from the latest blog posts in the directory.',
    expected_output='A 4-paragraph blog post formatted in markdown with engaging, informative, and accessible content, avoiding complex jargon.',
    agent=writer,
    output_file='blog-posts/new_post.md'  # The final blog post will be saved here
)

# Assemble a crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research, write],
    verbose=2
)

# Execute tasks
crew.kickoff()
```

## Available crewAI Tools

- **Error Handling**: All tools are built with error handling capabilities, allowing agents to gracefully manage exceptions and continue their tasks.
- **Caching Mechanism**: All tools support caching, enabling agents to efficiently reuse previously obtained results, reducing the load on external resources and speeding up the execution time. You can also define finer control over the caching mechanism using the `cache_function` attribute on the tool.

Here is a list of the available tools and their descriptions:

| Tool                        | Description                                                                                   |
| :-------------------------- | :-------------------------------------------------------------------------------------------- |
| **BrowserbaseLoadTool**     | A tool for interacting with and extracting data from web browsers.                            |
| **CodeDocsSearchTool**      | A RAG tool optimized for searching through code documentation and related technical documents. |
| **CodeInterpreterTool**     | A tool for interpreting python code.                                                          |
| **ComposioTool**            | Enables use of Composio tools.                                                                |
| **CSVSearchTool**           | A RAG tool designed for searching within CSV files, tailored to handle structured data.       |
| **DirectorySearchTool**     | A RAG tool for searching within directories, useful for navigating through file systems.      |
| **DOCXSearchTool**          | A RAG tool aimed at searching within DOCX documents, ideal for processing Word files.         |
| **DirectoryReadTool**       | Facilitates reading and processing of directory structures and their contents.                |
| **EXASearchTool**           | A tool designed for performing exhaustive searches across various data sources.               |
| **FileReadTool**            | Enables reading and extracting data from files, supporting various file formats.              |
| **FirecrawlSearchTool**     | A tool to search webpages using Firecrawl and return the results.                             |
| **FirecrawlCrawlWebsiteTool** | A tool for crawling webpages using Firecrawl.                                               |
| **FirecrawlScrapeWebsiteTool** | A tool for scraping webpages url using Firecrawl and returning its contents.               |
| **GithubSearchTool**        | A RAG tool for searching within GitHub repositories, useful for code and documentation search.|
| **SerperDevTool**           | A specialized tool for development purposes, with specific functionalities under development. |
| **TXTSearchTool**           | A RAG tool focused on searching within text (.txt) files, suitable for unstructured data.     |
| **JSONSearchTool**          | A RAG tool designed for searching within JSON files, catering to structured data handling.     |
| **LlamaIndexTool**          | Enables the use of LlamaIndex tools.                                                          |
| **MDXSearchTool**           | A RAG tool tailored for searching within Markdown (MDX) files, useful for documentation.      |
| **PDFSearchTool**           | A RAG tool aimed at searching within PDF documents, ideal for processing scanned documents.    |
| **PGSearchTool**            | A RAG tool optimized for searching within PostgreSQL databases, suitable for database queries. |
| **RagTool**                 | A general-purpose RAG tool capable of handling various data sources and types.                 |
| **ScrapeElementFromWebsiteTool** | Enables scraping specific elements from websites, useful for targeted data extraction.     |
| **ScrapeWebsiteTool**       | Facilitates scraping entire websites, ideal for comprehensive data collection.                 |
| **WebsiteSearchTool**       | A RAG tool for searching website content, optimized for web data extraction.                   |
| **XMLSearchTool**           | A RAG tool designed for searching within XML files, suitable for structured data formats.      |
| **YoutubeChannelSearchTool**| A RAG tool for searching within YouTube channels, useful for video content analysis.           |
| **YoutubeVideoSearchTool**  | A RAG tool aimed at searching within YouTube videos, ideal for video data extraction.          |

## Creating your own Tools

!!! example "Custom Tool Creation"
    Developers can craft custom tools tailored for their agent’s needs or utilize pre-built options:

To create your own crewAI tools you will need to install our extra tools package:

```bash
pip install 'crewai[tools]'
```

Once you do that there are two main ways for one to create a crewAI tool:
### Subclassing `BaseTool`

```python
from crewai_tools import BaseTool

class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = "Clear description for what this tool is useful for, your agent will need this information to use it."

    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "Result from custom tool"
```

### Utilizing the `tool` Decorator

```python
from crewai_tools import tool
@tool("Name of my tool")
def my_tool(question: str) -> str:
    """Clear description for what this tool is useful for, your agent will need this information to use it."""
    # Function logic here
    return "Result from your custom tool"
```

### Custom Caching Mechanism
!!! note "Caching"
    Tools can optionally implement a `cache_function` to fine-tune caching behavior. This function determines when to cache results based on specific conditions, offering granular control over caching logic.

```python
from crewai_tools import tool

@tool
def multiplication_tool(first_number: int, second_number: int) -> str:
    """Useful for when you need to multiply two numbers together."""
    return first_number * second_number

def cache_func(args, result):
    # In this case, we only cache the result if it's a multiple of 2
    cache = result % 2 == 0
    return cache

multiplication_tool.cache_function = cache_func

writer1 = Agent(
        role="Writer",
        goal="You write lessons of math for kids.",
        backstory="You're an expert in writing and you love to teach kids but you know nothing of math.",
        tools=[multiplication_tool],
        allow_delegation=False,
    )
    #...
```


## Conclusion
Tools are pivotal in extending the capabilities of CrewAI agents, enabling them to undertake a broad spectrum of tasks and collaborate effectively. When building solutions with CrewAI, leverage both custom and existing tools to empower your agents and enhance the AI ecosystem. Consider utilizing error handling, caching mechanisms, and the flexibility of tool arguments to optimize your agents' performance and capabilities.

---


# Training-Crew.md

---
title: crewAI Train
description: Learn how to train your crewAI agents by giving them feedback early on and get consistent results.
---

## Introduction
The training feature in CrewAI allows you to train your AI agents using the command-line interface (CLI). By running the command `crewai train -n <n_iterations>`, you can specify the number of iterations for the training process.

During training, CrewAI utilizes techniques to optimize the performance of your agents along with human feedback. This helps the agents improve their understanding, decision-making, and problem-solving abilities.

### Training Your Crew Using the CLI
To use the training feature, follow these steps:

1. Open your terminal or command prompt.
2. Navigate to the directory where your CrewAI project is located.
3. Run the following command:

```shell
crewai train -n <n_iterations>
```

### Training Your Crew Programmatically
To train your crew programmatically, use the following steps:

1. Define the number of iterations for training.
2. Specify the input parameters for the training process.
3. Execute the training command within a try-except block to handle potential errors.

```python
    n_iterations = 2
    inputs = {"topic": "CrewAI Training"}

    try:
        YourCrewName_Crew().crew().train(n_iterations= n_iterations, inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")
```

!!! note "Replace `<n_iterations>` with the desired number of training iterations. This determines how many times the agents will go through the training process."


### Key Points to Note:
- **Positive Integer Requirement:** Ensure that the number of iterations (`n_iterations`) is a positive integer. The code will raise a `ValueError` if this condition is not met.
- **Error Handling:** The code handles subprocess errors and unexpected exceptions, providing error messages to the user.

It is important to note that the training process may take some time, depending on the complexity of your agents and will also require your feedback on each iteration.

Once the training is complete, your agents will be equipped with enhanced capabilities and knowledge, ready to tackle complex tasks and provide more consistent and valuable insights.

Remember to regularly update and retrain your agents to ensure they stay up-to-date with the latest information and advancements in the field.

Happy training with CrewAI!

---


# Using-LangChain-Tools.md

---
title: Using LangChain Tools
description: Learn how to integrate LangChain tools with CrewAI agents to enhance search-based queries and more.
---

## Using LangChain Tools
!!! info "LangChain Integration"
    CrewAI seamlessly integrates with LangChain’s comprehensive toolkit for search-based queries and more, here are the available built-in tools that are offered by Langchain [LangChain Toolkit](https://python.langchain.com/docs/integrations/tools/)

```python
from crewai import Agent
from langchain.agents import Tool
from langchain.utilities import GoogleSerperAPIWrapper

# Setup API keys
os.environ["SERPER_API_KEY"] = "Your Key"

search = GoogleSerperAPIWrapper()

# Create and assign the search tool to an agent
serper_tool = Tool(
  name="Intermediate Answer",
  func=search.run,
  description="Useful for search-based queries",
)

agent = Agent(
  role='Research Analyst',
  goal='Provide up-to-date market analysis',
  backstory='An expert analyst with a keen eye for market trends.',
  tools=[serper_tool]
)

# rest of the code ...
```

## Conclusion
Tools are pivotal in extending the capabilities of CrewAI agents, enabling them to undertake a broad spectrum of tasks and collaborate effectively. When building solutions with CrewAI, leverage both custom and existing tools to empower your agents and enhance the AI ecosystem. Consider utilizing error handling, caching mechanisms, and the flexibility of tool arguments to optimize your agents' performance and capabilities.

---


# Using-LlamaIndex-Tools.md

---
title: Using LlamaIndex Tools
description: Learn how to integrate LlamaIndex tools with CrewAI agents to enhance search-based queries and more.
---

## Using LlamaIndex Tools

!!! info "LlamaIndex Integration"
    CrewAI seamlessly integrates with LlamaIndex’s comprehensive toolkit for RAG (Retrieval-Augmented Generation) and agentic pipelines, enabling advanced search-based queries and more. Here are the available built-in tools offered by LlamaIndex.

```python
from crewai import Agent
from crewai_tools import LlamaIndexTool

# Example 1: Initialize from FunctionTool
from llama_index.core.tools import FunctionTool

your_python_function = lambda ...: ...
og_tool = FunctionTool.from_defaults(your_python_function, name="<name>", description='<description>')
tool = LlamaIndexTool.from_tool(og_tool)

# Example 2: Initialize from LlamaHub Tools
from llama_index.tools.wolfram_alpha import WolframAlphaToolSpec
wolfram_spec = WolframAlphaToolSpec(app_id="<app_id>")
wolfram_tools = wolfram_spec.to_tool_list()
tools = [LlamaIndexTool.from_tool(t) for t in wolfram_tools]

# Example 3: Initialize Tool from a LlamaIndex Query Engine
query_engine = index.as_query_engine()
query_tool = LlamaIndexTool.from_query_engine(
    query_engine,
    name="Uber 2019 10K Query Tool",
    description="Use this tool to lookup the 2019 Uber 10K Annual Report"
)

# Create and assign the tools to an agent
agent = Agent(
  role='Research Analyst',
  goal='Provide up-to-date market analysis',
  backstory='An expert analyst with a keen eye for market trends.',
  tools=[tool, *tools, query_tool]
)

# rest of the code ...
```

## Steps to Get Started

To effectively use the LlamaIndexTool, follow these steps:

1. **Package Installation**: Confirm that the `crewai[tools]` package is installed in your Python environment.

    ```shell
    pip install 'crewai[tools]'
    ```

2. **Install and Use LlamaIndex**: Follow LlamaIndex documentation [LlamaIndex Documentation](https://docs.llamaindex.ai/) to set up a RAG/agent pipeline.

---


# Installing-CrewAI.md

---
title: Installing crewAI
description: A comprehensive guide to installing crewAI and its dependencies, including the latest updates and installation methods.
---

# Installing crewAI

Welcome to crewAI! This guide will walk you through the installation process for crewAI and its dependencies. crewAI is a flexible and powerful AI framework that enables you to create and manage AI agents, tools, and tasks efficiently. Let's get started!

## Installation

To install crewAI, you need to have Python >=3.10 and <=3.13 installed on your system:

```shell
# Install the main crewAI package
pip install crewai

# Install the main crewAI package and the tools package
# that includes a series of helpful tools for your agents
pip install 'crewai[tools]'

# Alternatively, you can also use:
pip install crewai crewai-tools
```

---


# Start-a-New-CrewAI-Project-Template-Method.md

---
title: Starting a New CrewAI Project - Using Template
description: A comprehensive guide to starting a new CrewAI project, including the latest updates and project setup methods.
---

# Starting Your CrewAI Project

Welcome to the ultimate guide for starting a new CrewAI project. This document will walk you through the steps to create, customize, and run your CrewAI project, ensuring you have everything you need to get started.

Beforre we start there are a couple of things to note:

1. CrewAI is a Python package and requires Python >=3.10 and <=3.13 to run.
2. The preferred way of setting up CrewAI is using the `crewai create` command.This will create a new project folder and install a skeleton template for you to work on.

## Prerequisites

Before getting started with CrewAI, make sure that you have installed it via pip:

```shell
$ pip install crewai crewai-tools
```

### Virtual Environments
It is highly recommended that you use virtual environments to ensure that your CrewAI project is isolated from other projects and dependencies. Virtual environments provide a clean, separate workspace for each project, preventing conflicts between different versions of packages and libraries. This isolation is crucial for maintaining consistency and reproducibility in your development process. You have multiple options for setting up virtual environments depending on your operating system and Python version:

1. Use venv (Python's built-in virtual environment tool):
   venv is included with Python 3.3 and later, making it a convenient choice for many developers. It's lightweight and easy to use, perfect for simple project setups.

   To set up virtual environments with venv, refer to the official [Python documentation](https://docs.python.org/3/tutorial/venv.html).

2. Use Conda (A Python virtual environment manager):
   Conda is an open-source package manager and environment management system for Python. It's widely used by data scientists, developers, and researchers to manage dependencies and environments in a reproducible way.

   To set up virtual environments with Conda, refer to the official [Conda documentation](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html).

3. Use Poetry (A Python package manager and dependency management tool):
   Poetry is an open-source Python package manager that simplifies the installation of packages and their dependencies. Poetry offers a convenient way to manage virtual environments and dependencies.
   Poetry is CrewAI's prefered tool for package / dependancy management in CrewAI.

### Code IDEs

Most users of CrewAI a Code Editor / Integrated Development Environment (IDE) for building there Crews. You can use any code IDE of your choice. Seee below for some popular options for Code Editors / Integrated Development Environments (IDE):

- [Visual Studio Code](https://code.visualstudio.com/) - Most popular
- [PyCharm](https://www.jetbrains.com/pycharm/)
- [Cursor AI](https://cursor.com)

Pick one that suits your style and needs.

## Creating a New Project
In this example we will be using Venv as our virtual environment manager.

To setup a virtual environment, run the following CLI command:

```shell
$ python3 -m venv <venv-name>
```

Activate your virtual environment by running the following CLI command:

```shell
$ source <venv-name>/bin/activate
```

Now, to create a new CrewAI project, run the following CLI command:

```shell
$ crewai create <project_name>
```

This command will create a new project folder with the following structure:

```shell
my_project/
├── .gitignore
├── pyproject.toml
├── README.md
└── src/
    └── my_project/
        ├── __init__.py
        ├── main.py
        ├── crew.py
        ├── tools/
        │   ├── custom_tool.py
        │   └── __init__.py
        └── config/
            ├── agents.yaml
            └── tasks.yaml
```

You can now start developing your project by editing the files in the `src/my_project` folder. The `main.py` file is the entry point of your project, and the `crew.py` file is where you define your agents and tasks.

## Customizing Your Project

To customize your project, you can:
- Modify `src/my_project/config/agents.yaml` to define your agents.
- Modify `src/my_project/config/tasks.yaml` to define your tasks.
- Modify `src/my_project/crew.py` to add your own logic, tools, and specific arguments.
- Modify `src/my_project/main.py` to add custom inputs for your agents and tasks.
- Add your environment variables into the `.env` file.

### Example: Defining Agents and Tasks

#### agents.yaml

```yaml
researcher:
  role: >
    Job Candidate Researcher
  goal: >
    Find potential candidates for the job
  backstory: >
    You are adept at finding the right candidates by exploring various online
    resources. Your skill in identifying suitable candidates ensures the best
    match for job positions.
```

#### tasks.yaml

```yaml
research_candidates_task:
  description: >
    Conduct thorough research to find potential candidates for the specified job.
    Utilize various online resources and databases to gather a comprehensive list of potential candidates.
    Ensure that the candidates meet the job requirements provided.

    Job Requirements:
    {job_requirements}
  expected_output: >
    A list of 10 potential candidates with their contact information and brief profiles highlighting their suitability.
  agent: researcher # THIS NEEDS TO MATCH THE AGENT NAME IN THE AGENTS.YAML FILE AND THE AGENT DEFINED IN THE Crew.PY FILE
  context: # THESE NEED TO MATCH THE TASK NAMES DEFINED ABOVE AND THE TASKS.YAML FILE AND THE TASK DEFINED IN THE Crew.PY FILE
    - researcher
```

### Referencing Variables:
Your defined functions with the same name will be used. For example, you can reference the agent for specific tasks from task.yaml file. Ensure your annotated agent and function name is the same otherwise your task wont recognize the reference properly.

#### Example References
agent.yaml
```yaml
email_summarizer:
    role: >
      Email Summarizer
    goal: >
      Summarize emails into a concise and clear summary
    backstory: >
      You will create a 5 bullet point summary of the report
    llm: mixtal_llm
```

task.yaml
```yaml
email_summarizer_task:
    description: >
      Summarize the email into a 5 bullet point summary
    expected_output: >
      A 5 bullet point summary of the email
    agent: email_summarizer
    context:
      - reporting_task
      - research_task
```

Use the annotations are used to properly reference the agent and task in the crew.py file.

### Annotations include:
* @agent
* @task
* @crew
* @llm
* @tool
* @callback
* @output_json
* @output_pydantic
* @cache_handler


crew.py
```py
...
    @llm
    def mixtal_llm(self):
        return ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

    @agent
    def email_summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config["email_summarizer"],
        )
    ## ...other tasks defined
    @task
    def email_summarizer_task(self) -> Task:
        return Task(
            config=self.tasks_config["email_summarizer_task"],
        )
...
```



## Installing Dependencies

To install the dependencies for your project, you can use Poetry. First, navigate to your project directory:

```shell
$ cd my_project
$ poetry lock
$ poetry install
```

This will install the dependencies specified in the `pyproject.toml` file.

## Interpolating Variables

Any variable interpolated in your `agents.yaml` and `tasks.yaml` files like `{variable}` will be replaced by the value of the variable in the `main.py` file.

#### agents.yaml

```yaml
research_task:
  description: >
    Conduct a thorough research about the customer and competitors in the context
    of {customer_domain}.
    Make sure you find any interesting and relevant information given the
    current year is 2024.
  expected_output: >
    A complete report on the customer and their customers and competitors,
    including their demographics, preferences, market positioning and audience engagement.
```

#### main.py

```python
# main.py
def run():
    inputs = {
        "customer_domain": "crewai.com"
    }
    MyProjectCrew(inputs).crew().kickoff(inputs=inputs)
```

## Running Your Project

To run your project, use the following command:

```shell
$ poetry run my_project
```

This will initialize your crew of AI agents and begin task execution as defined in your configuration in the `main.py` file.

## Deploying Your Project

The easiest way to deploy your crew is through [CrewAI+](https://www.crewai.com/crewaiplus), where you can deploy your crew in a few clicks.


---


# AgentOps-Observability.md

---
title: Agent Monitoring with AgentOps
description: Understanding and logging your agent performance with AgentOps.
---

# Intro
Observability is a key aspect of developing and deploying conversational AI agents. It allows developers to understand how their agents are performing, how their agents are interacting with users, and how their agents use external tools and APIs. AgentOps is a product independent of CrewAI that provides a comprehensive observability solution for agents.

## AgentOps

[AgentOps](https://agentops.ai/?=crew) provides session replays, metrics, and monitoring for agents.

At a high level, AgentOps gives you the ability to monitor cost, token usage, latency, agent failures, session-wide statistics, and more. For more info, check out the [AgentOps Repo](https://github.com/AgentOps-AI/agentops).

### Overview
AgentOps provides monitoring for agents in development and production. It provides a dashboard for tracking agent performance, session replays, and custom reporting.

Additionally, AgentOps provides session drilldowns for viewing Crew agent interactions, LLM calls, and tool usage in real-time. This feature is useful for debugging and understanding how agents interact with users as well as other agents.

![Overview of a select series of agent session runs](..%2Fassets%2Fagentops-overview.png)
![Overview of session drilldowns for examining agent runs](..%2Fassets%2Fagentops-session.png)
![Viewing a step-by-step agent replay execution graph](..%2Fassets%2Fagentops-replay.png)

### Features
- **LLM Cost Management and Tracking**: Track spend with foundation model providers.
- **Replay Analytics**: Watch step-by-step agent execution graphs.
- **Recursive Thought Detection**: Identify when agents fall into infinite loops.
- **Custom Reporting**: Create custom analytics on agent performance.
- **Analytics Dashboard**: Monitor high-level statistics about agents in development and production.
- **Public Model Testing**: Test your agents against benchmarks and leaderboards.
- **Custom Tests**: Run your agents against domain-specific tests.
- **Time Travel Debugging**: Restart your sessions from checkpoints.
- **Compliance and Security**: Create audit logs and detect potential threats such as profanity and PII leaks.
- **Prompt Injection Detection**: Identify potential code injection and secret leaks.

### Using AgentOps

1. **Create an API Key:**
   Create a user API key here: [Create API Key](app.agentops.ai/account)

2. **Configure Your Environment:**
   Add your API key to your environment variables

   ```bash
   AGENTOPS_API_KEY=<YOUR_AGENTOPS_API_KEY>
   ```

3. **Install AgentOps:**
   Install AgentOps with:
   ```bash
   pip install crewai[agentops]
   ```
   or
   ```bash
   pip install agentops
   ```

   Before using `Crew` in your script, include these lines:

   ```python
   import agentops
   agentops.init()
   ```

   This will initiate an AgentOps session as well as automatically track Crew agents. For further info on how to outfit more complex agentic systems, check out the [AgentOps documentation](https://docs.agentops.ai) or join the [Discord](https://discord.gg/j4f3KbeH).

### Crew + AgentOps Examples
- [Job Posting](https://github.com/joaomdmoura/crewAI-examples/tree/main/job-posting)
- [Markdown Validator](https://github.com/joaomdmoura/crewAI-examples/tree/main/markdown_validator)
- [Instagram Post](https://github.com/joaomdmoura/crewAI-examples/tree/main/instagram_post)

### Further Information

To get started, create an [AgentOps account](https://agentops.ai/?=crew).

For feature requests or bug reports, please reach out to the AgentOps team on the [AgentOps Repo](https://github.com/AgentOps-AI/agentops).

#### Extra links

<a href="https://twitter.com/agentopsai/">🐦 Twitter</a>
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
<a href="https://discord.gg/JHPt4C7r">📢 Discord</a>
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
<a href="https://app.agentops.ai/?=crew">🖇️ AgentOps Dashboard</a>
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
<a href="https://docs.agentops.ai/introduction">📙 Documentation</a>

---


# Coding-Agents.md

---
title: Coding Agents
description: Learn how to enable your crewAI Agents to write and execute code, and explore advanced features for enhanced functionality.
---

## Introduction

crewAI Agents now have the powerful ability to write and execute code, significantly enhancing their problem-solving capabilities. This feature is particularly useful for tasks that require computational or programmatic solutions.

## Enabling Code Execution

To enable code execution for an agent, set the `allow_code_execution` parameter to `True` when creating the agent. Here's an example:

```python
from crewai import Agent

coding_agent = Agent(
    role="Senior Python Developer",
    goal="Craft well-designed and thought-out code",
    backstory="You are a senior Python developer with extensive experience in software architecture and best practices.",
    allow_code_execution=True
)
```

## Important Considerations

1. **Model Selection**: It is strongly recommended to use more capable models like Claude 3.5 Sonnet and GPT-4 when enabling code execution. These models have a better understanding of programming concepts and are more likely to generate correct and efficient code.

2. **Error Handling**: The code execution feature includes error handling. If executed code raises an exception, the agent will receive the error message and can attempt to correct the code or provide alternative solutions.

3. **Dependencies**: To use the code execution feature, you need to install the `crewai_tools` package. If not installed, the agent will log an info message: "Coding tools not available. Install crewai_tools."

## Code Execution Process

When an agent with code execution enabled encounters a task requiring programming:

1. The agent analyzes the task and determines that code execution is necessary.
2. It formulates the Python code needed to solve the problem.
3. The code is sent to the internal code execution tool (`CodeInterpreterTool`).
4. The tool executes the code in a controlled environment and returns the result.
5. The agent interprets the result and incorporates it into its response or uses it for further problem-solving.

## Example Usage

Here's a detailed example of creating an agent with code execution capabilities and using it in a task:

```python
from crewai import Agent, Task, Crew

# Create an agent with code execution enabled
coding_agent = Agent(
    role="Python Data Analyst",
    goal="Analyze data and provide insights using Python",
    backstory="You are an experienced data analyst with strong Python skills.",
    allow_code_execution=True
)

# Create a task that requires code execution
data_analysis_task = Task(
    description="Analyze the given dataset and calculate the average age of participants.",
    agent=coding_agent
)

# Create a crew and add the task
analysis_crew = Crew(
    agents=[coding_agent],
    tasks=[data_analysis_task]
)

# Execute the crew
result = analysis_crew.kickoff()

print(result)
```

In this example, the `coding_agent` can write and execute Python code to perform data analysis tasks.

---


# Conditional-Tasks.md

---
title: Conditional Tasks
description: Learn how to use conditional tasks in a crewAI kickoff
---

## Introduction

Conditional Tasks in crewAI allow for dynamic workflow adaptation based on the outcomes of previous tasks. This powerful feature enables crews to make decisions and execute tasks selectively, enhancing the flexibility and efficiency of your AI-driven processes.

```python
from typing import List

from pydantic import BaseModel
from crewai import Agent, Crew
from crewai.tasks.conditional_task import ConditionalTask
from crewai.tasks.task_output import TaskOutput
from crewai.task import Task
from crewai_tools import SerperDevTool


# Define a condition function for the conditional task
# if false task will be skipped, true, then execute task
def is_data_missing(output: TaskOutput) -> bool:
    return len(output.pydantic.events) < 10: # this will skip this task

# Define the agents
data_fetcher_agent = Agent(
    role="Data Fetcher",
    goal="Fetch data online using Serper tool",
    backstory="Backstory 1",
    verbose=True,
    tools=[SerperDevTool()],
)

data_processor_agent = Agent(
    role="Data Processor",
    goal="Process fetched data",
    backstory="Backstory 2",
    verbose=True,
)

summary_generator_agent = Agent(
    role="Summary Generator",
    goal="Generate summary from fetched data",
    backstory="Backstory 3",
    verbose=True,
)


class EventOutput(BaseModel):
    events: List[str]


task1 = Task(
    description="Fetch data about events in San Francisco using Serper tool",
    expected_output="List of 10 things to do in SF this week",
    agent=data_fetcher_agent,
    output_pydantic=EventOutput,
)

conditional_task = ConditionalTask(
    description="""
        Check if data is missing. If we have less than 10 events,
        fetch more events using Serper tool so that
        we have a total of 10 events in SF this week..
        """,
    expected_output="List of 10 Things to do in SF this week ",
    condition=is_data_missing,
    agent=data_processor_agent,
)

task3 = Task(
    description="Generate summary of events in San Francisco from fetched data",
    expected_output="summary_generated",
    agent=summary_generator_agent,
)

# Create a crew with the tasks
crew = Crew(
    agents=[data_fetcher_agent, data_processor_agent, summary_generator_agent],
    tasks=[task1, conditional_task, task3],
    verbose=2,
)

result = crew.kickoff()
print("results", result)
```


---


# Create-Custom-Tools.md

---
title: Creating and Utilizing Tools in crewAI
description: Comprehensive guide on crafting, using, and managing custom tools within the crewAI framework, including new functionalities and error handling.
---

## Creating and Utilizing Tools in crewAI
This guide provides detailed instructions on creating custom tools for the crewAI framework and how to efficiently manage and utilize these tools, incorporating the latest functionalities such as tool delegation, error handling, and dynamic tool calling. It also highlights the importance of collaboration tools, enabling agents to perform a wide range of actions.

### Prerequisites

Before creating your own tools, ensure you have the crewAI extra tools package installed:

```bash
pip install 'crewai[tools]'
```

### Subclassing `BaseTool`

To create a personalized tool, inherit from `BaseTool` and define the necessary attributes and the `_run` method.

```python
from crewai_tools import BaseTool

class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = "What this tool does. It's vital for effective utilization."

    def _run(self, argument: str) -> str:
        # Your tool's logic here
        return "Tool's result"
```

### Using the `tool` Decorator

Alternatively, you can use the tool decorator `@tool`. This approach allows you to define the tool's attributes and functionality directly within a function, offering a concise and efficient way to create specialized tools tailored to your needs.

```python
from crewai_tools import tool

@tool("Tool Name")
def my_simple_tool(question: str) -> str:
    """Tool description for clarity."""
    # Tool logic here
    return "Tool output"
```

### Defining a Cache Function for the Tool

To optimize tool performance with caching, define custom caching strategies using the `cache_function` attribute.

```python
@tool("Tool with Caching")
def cached_tool(argument: str) -> str:
    """Tool functionality description."""
    return "Cacheable result"

def my_cache_strategy(arguments: dict, result: str) -> bool:
    # Define custom caching logic
    return True if some_condition else False

cached_tool.cache_function = my_cache_strategy
```

By adhering to these guidelines and incorporating new functionalities and collaboration tools into your tool creation and management processes, you can leverage the full capabilities of the crewAI framework, enhancing both the development experience and the efficiency of your AI agents.

---


# Customize-Prompts.md

---
title: Initial Support to Bring Your Own Prompts in CrewAI
description: Enhancing customization and internationalization by allowing users to bring their own prompts in CrewAI.

---

# Initial Support to Bring Your Own Prompts in CrewAI

CrewAI now supports the ability to bring your own prompts, enabling extensive customization and internationalization. This feature allows users to tailor the inner workings of their agents to better suit specific needs, including support for multiple languages.

## Internationalization and Customization Support

### Custom Prompts with `prompt_file`

The `prompt_file` attribute facilitates full customization of the agent prompts, enhancing the global usability of CrewAI. Users can specify their prompt templates, ensuring that the agents communicate in a manner that aligns with specific project requirements or language preferences.

#### Example of a Custom Prompt File

The custom prompts can be defined in a JSON file, similar to the example provided [here](https://github.com/joaomdmoura/crewAI/blob/main/src/crewai/translations/en.json).

### Supported Languages

CrewAI's custom prompt support includes internationalization, allowing prompts to be written in different languages. This is particularly useful for global teams or projects that require multilingual support.

## How to Use the `prompt_file` Attribute

To utilize the `prompt_file` attribute, include it in your crew definition. Below is an example demonstrating how to set up agents and tasks with custom prompts.

### Example

```python
import os
from crewai import Agent, Task, Crew

# Define your agents
researcher = Agent(
    role="Researcher",
    goal="Make the best research and analysis on content about AI and AI agents",
    backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
    allow_delegation=False,
)

writer = Agent(
    role="Senior Writer",
    goal="Write the best content about AI and AI agents.",
    backstory="You're a senior writer, specialized in technology, software engineering, AI and startups. You work as a freelancer and are now working on writing content for a new customer.",
    allow_delegation=False,
)

# Define your tasks
tasks = [
    Task(
        description="Say Hi",
        expected_output="The word: Hi",
        agent=researcher,
    )
]

# Instantiate your crew with custom prompts
crew = Crew(
    agents=[researcher],
    tasks=tasks,
    prompt_file="prompt.json",  # Path to your custom prompt file
)

# Get your crew to work!
crew.kickoff()
```

## Advanced Customization Features

### `language` Attribute

In addition to `prompt_file`, the `language` attribute can be used to specify the language for the agent's prompts. This ensures that the prompts are generated in the desired language, further enhancing the internationalization capabilities of CrewAI.

### Creating Custom Prompt Files

Custom prompt files should be structured in JSON format and include all necessary prompt templates. Below is a simplified example of a prompt JSON file:

```json
{
    "system": "You are a system template.",
    "prompt": "Here is your prompt template.",
    "response": "Here is your response template."
}
```

### Benefits of Custom Prompts

- **Enhanced Flexibility**: Tailor agent communication to specific project needs.
- **Improved Usability**: Supports multiple languages, making it suitable for global projects.
- **Consistency**: Ensures uniform prompt structures across different agents and tasks.

By incorporating these updates, CrewAI provides users with the ability to fully customize and internationalize their agent prompts, making the platform more versatile and user-friendly.


---


# Customizing-Agents.md

---
title: Customizing Agents in CrewAI
description: A comprehensive guide to tailoring agents for specific roles, tasks, and advanced customizations within the CrewAI framework.
---

## Customizable Attributes
Crafting an efficient CrewAI team hinges on the ability to dynamically tailor your AI agents to meet the unique requirements of any project. This section covers the foundational attributes you can customize.

### Key Attributes for Customization
- **Role**: Specifies the agent's job within the crew, such as 'Analyst' or 'Customer Service Rep'.
- **Goal**: Defines what the agent aims to achieve, in alignment with its role and the overarching objectives of the crew.
- **Backstory**: Provides depth to the agent's persona, enriching its motivations and engagements within the crew.
- **Tools** *(Optional)*: Represents the capabilities or methods the agent uses to perform tasks, from simple functions to intricate integrations.
- **Cache** *(Optional)*: Determines whether the agent should use a cache for tool usage.
- **Max RPM**: Sets the maximum number of requests per minute (`max_rpm`). This attribute is optional and can be set to `None` for no limit, allowing for unlimited queries to external services if needed.
- **Verbose** *(Optional)*: Enables detailed logging of an agent's actions, useful for debugging and optimization. Specifically, it provides insights into agent execution processes, aiding in the optimization of performance.
- **Allow Delegation** *(Optional)*: `allow_delegation` controls whether the agent is allowed to delegate tasks to other agents.
- **Max Iter** *(Optional)*: The `max_iter` attribute allows users to define the maximum number of iterations an agent can perform for a single task, preventing infinite loops or excessively long executions. The default value is set to 25, providing a balance between thoroughness and efficiency. Once the agent approaches this number, it will try its best to give a good answer.
- **Max Execution Time** *(Optional)*: `max_execution_time` Sets the maximum execution time for an agent to complete a task.
- **System Template** *(Optional)*: `system_template` defines the system format for the agent.
- **Prompt Template** *(Optional)*: `prompt_template` defines the prompt format for the agent.
- **Response Template** *(Optional)*: `response_template` defines the response format for the agent.

## Advanced Customization Options
Beyond the basic attributes, CrewAI allows for deeper customization to enhance an agent's behavior and capabilities significantly.

### Language Model Customization
Agents can be customized with specific language models (`llm`) and function-calling language models (`function_calling_llm`), offering advanced control over their processing and decision-making abilities. It's important to note that setting the `function_calling_llm` allows for overriding the default crew function-calling language model, providing a greater degree of customization.

## Performance and Debugging Settings
Adjusting an agent's performance and monitoring its operations are crucial for efficient task execution.

### Verbose Mode and RPM Limit
- **Verbose Mode**: Enables detailed logging of an agent's actions, useful for debugging and optimization. Specifically, it provides insights into agent execution processes, aiding in the optimization of performance.
- **RPM Limit**: Sets the maximum number of requests per minute (`max_rpm`). This attribute is optional and can be set to `None` for no limit, allowing for unlimited queries to external services if needed.

### Maximum Iterations for Task Execution
The `max_iter` attribute allows users to define the maximum number of iterations an agent can perform for a single task, preventing infinite loops or excessively long executions. The default value is set to 25, providing a balance between thoroughness and efficiency. Once the agent approaches this number, it will try its best to give a good answer.

## Customizing Agents and Tools
Agents are customized by defining their attributes and tools during initialization. Tools are critical for an agent's functionality, enabling them to perform specialized tasks. The `tools` attribute should be an array of tools the agent can utilize, and it's initialized as an empty list by default. Tools can be added or modified post-agent initialization to adapt to new requirements.

```shell
pip install 'crewai[tools]'
```

### Example: Assigning Tools to an Agent
```python
import os
from crewai import Agent
from crewai_tools import SerperDevTool

# Set API keys for tool initialization
os.environ["OPENAI_API_KEY"] = "Your Key"
os.environ["SERPER_API_KEY"] = "Your Key"

# Initialize a search tool
search_tool = SerperDevTool()

# Initialize the agent with advanced options
agent = Agent(
  role='Research Analyst',
  goal='Provide up-to-date market analysis',
  backstory='An expert analyst with a keen eye for market trends.',
  tools=[search_tool],
  memory=True, # Enable memory
  verbose=True,
  max_rpm=None, # No limit on requests per minute
  max_iter=25, # Default value for maximum iterations
  allow_delegation=False
)
```

## Delegation and Autonomy
Controlling an agent's ability to delegate tasks or ask questions is vital for tailoring its autonomy and collaborative dynamics within the CrewAI framework. By default, the `allow_delegation` attribute is set to `True`, enabling agents to seek assistance or delegate tasks as needed. This default behavior promotes collaborative problem-solving and efficiency within the CrewAI ecosystem. If needed, delegation can be disabled to suit specific operational requirements.

### Example: Disabling Delegation for an Agent
```python
agent = Agent(
  role='Content Writer',
  goal='Write engaging content on market trends',
  backstory='A seasoned writer with expertise in market analysis.',
  allow_delegation=False # Disabling delegation
)
```

## Conclusion
Customizing agents in CrewAI by setting their roles, goals, backstories, and tools, alongside advanced options like language model customization, memory, performance settings, and delegation preferences, equips a nuanced and capable AI team ready for complex challenges.

---


# Force-Tool-Ouput-as-Result.md

---
title: Forcing Tool Output as Result
description: Learn how to force tool output as the result in of an Agent's task in crewAI.
---

## Introduction
In CrewAI, you can force the output of a tool as the result of an agent's task. This feature is useful when you want to ensure that the tool output is captured and returned as the task result, and avoid the agent modifying the output during the task execution.

## Forcing Tool Output as Result
To force the tool output as the result of an agent's task, you can set the `result_as_answer` parameter to `True` when creating the agent. This parameter ensures that the tool output is captured and returned as the task result, without any modifications by the agent.

Here's an example of how to force the tool output as the result of an agent's task:

```python
# ...
# Define a custom tool that returns the result as the answer
coding_agent =Agent(
        role="Data Scientist",
        goal="Product amazing reports on AI",
        backstory="You work with data and AI",
        tools=[MyCustomTool(result_as_answer=True)],
    )
# ...
```

### Workflow in Action

1. **Task Execution**: The agent executes the task using the tool provided.
2. **Tool Output**: The tool generates the output, which is captured as the task result.
3. **Agent Interaction**: The agent my reflect and take learnings from the tool but the output is not modified.
4. **Result Return**: The tool output is returned as the task result without any modifications.


---


# Hierarchical.md

---
title: Implementing the Hierarchical Process in CrewAI
description: A comprehensive guide to understanding and applying the hierarchical process within your CrewAI projects, updated to reflect the latest coding practices and functionalities.
---

## Introduction
The hierarchical process in CrewAI introduces a structured approach to task management, simulating traditional organizational hierarchies for efficient task delegation and execution. This systematic workflow enhances project outcomes by ensuring tasks are handled with optimal efficiency and accuracy.

!!! note "Complexity and Efficiency"
    The hierarchical process is designed to leverage advanced models like GPT-4, optimizing token usage while handling complex tasks with greater efficiency.

## Hierarchical Process Overview
By default, tasks in CrewAI are managed through a sequential process. However, adopting a hierarchical approach allows for a clear hierarchy in task management, where a 'manager' agent coordinates the workflow, delegates tasks, and validates outcomes for streamlined and effective execution. This manager agent can now be either automatically created by CrewAI or explicitly set by the user.

### Key Features
- **Task Delegation**: A manager agent allocates tasks among crew members based on their roles and capabilities.
- **Result Validation**: The manager evaluates outcomes to ensure they meet the required standards.
- **Efficient Workflow**: Emulates corporate structures, providing an organized approach to task management.

## Implementing the Hierarchical Process
To utilize the hierarchical process, it's essential to explicitly set the process attribute to `Process.hierarchical`, as the default behavior is `Process.sequential`. Define a crew with a designated manager and establish a clear chain of command.

!!! note "Tools and Agent Assignment"
    Assign tools at the agent level to facilitate task delegation and execution by the designated agents under the manager's guidance. Tools can also be specified at the task level for precise control over tool availability during task execution.

!!! note "Manager LLM Requirement"
    Configuring the `manager_llm` parameter is crucial for the hierarchical process. The system requires a manager LLM to be set up for proper function, ensuring tailored decision-making.

```python
from langchain_openai import ChatOpenAI
from crewai import Crew, Process, Agent

# Agents are defined with attributes for backstory, cache, and verbose mode
researcher = Agent(
    role='Researcher',
    goal='Conduct in-depth analysis',
    backstory='Experienced data analyst with a knack for uncovering hidden trends.',
    cache=True,
    verbose=False,
    # tools=[]  # This can be optionally specified; defaults to an empty list
)
writer = Agent(
    role='Writer',
    goal='Create engaging content',
    backstory='Creative writer passionate about storytelling in technical domains.',
    cache=True,
    verbose=False,
    # tools=[]  # Optionally specify tools; defaults to an empty list
)

# Establishing the crew with a hierarchical process and additional configurations
project_crew = Crew(
    tasks=[...],  # Tasks to be delegated and executed under the manager's supervision
    agents=[researcher, writer],
    manager_llm=ChatOpenAI(temperature=0, model="gpt-4"),  # Mandatory if manager_agent is not set
    process=Process.hierarchical,  # Specifies the hierarchical management approach
    memory=True,  # Enable memory usage for enhanced task execution
    manager_agent=None,  # Optional: explicitly set a specific agent as manager instead of the manager_llm
)
```

### Workflow in Action
1. **Task Assignment**: The manager assigns tasks strategically, considering each agent's capabilities and available tools.
2. **Execution and Review**: Agents complete their tasks with the option for asynchronous execution and callback functions for streamlined workflows.
3. **Sequential Task Progression**: Despite being a hierarchical process, tasks follow a logical order for smooth progression, facilitated by the manager's oversight.

## Conclusion
Adopting the hierarchical process in CrewAI, with the correct configurations and understanding of the system's capabilities, facilitates an organized and efficient approach to project management. Utilize the advanced features and customizations to tailor the workflow to your specific needs, ensuring optimal task execution and project success.

---


# Human-Input-on-Execution.md

---
title: Human Input on Execution
description: Integrating CrewAI with human input during execution in complex decision-making processes and leveraging the full capabilities of the agent's attributes and tools.
---

# Human Input in Agent Execution

Human input is critical in several agent execution scenarios, allowing agents to request additional information or clarification when necessary. This feature is especially useful in complex decision-making processes or when agents require more details to complete a task effectively.

## Using Human Input with CrewAI

To integrate human input into agent execution, set the `human_input` flag in the task definition. When enabled, the agent prompts the user for input before delivering its final answer. This input can provide extra context, clarify ambiguities, or validate the agent's output.

### Example:

```shell
pip install crewai
```

```python
import os
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool

os.environ["SERPER_API_KEY"] = "Your Key"  # serper.dev API key
os.environ["OPENAI_API_KEY"] = "Your Key"

# Loading Tools
search_tool = SerperDevTool()

# Define your agents with roles, goals, tools, and additional attributes
researcher = Agent(
    role='Senior Research Analyst',
    goal='Uncover cutting-edge developments in AI and data science',
    backstory=(
        "You are a Senior Research Analyst at a leading tech think tank. "
        "Your expertise lies in identifying emerging trends and technologies in AI and data science. "
        "You have a knack for dissecting complex data and presenting actionable insights."
    ),
    verbose=True,
    allow_delegation=False,
    tools=[search_tool]
)
writer = Agent(
    role='Tech Content Strategist',
    goal='Craft compelling content on tech advancements',
    backstory=(
        "You are a renowned Tech Content Strategist, known for your insightful and engaging articles on technology and innovation. "
        "With a deep understanding of the tech industry, you transform complex concepts into compelling narratives."
    ),
    verbose=True,
    allow_delegation=True,
    tools=[search_tool],
    cache=False,  # Disable cache for this agent
)

# Create tasks for your agents
task1 = Task(
    description=(
        "Conduct a comprehensive analysis of the latest advancements in AI in 2024. "
        "Identify key trends, breakthrough technologies, and potential industry impacts. "
        "Compile your findings in a detailed report. "
        "Make sure to check with a human if the draft is good before finalizing your answer."
    ),
    expected_output='A comprehensive full report on the latest AI advancements in 2024, leave nothing out',
    agent=researcher,
    human_input=True
)

task2 = Task(
    description=(
        "Using the insights from the researcher\'s report, develop an engaging blog post that highlights the most significant AI advancements. "
        "Your post should be informative yet accessible, catering to a tech-savvy audience. "
        "Aim for a narrative that captures the essence of these breakthroughs and their implications for the future."
    ),
    expected_output='A compelling 3 paragraphs blog post formatted as markdown about the latest AI advancements in 2024',
    agent=writer
)

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=2,
    memory=True,
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)
```

---


# Kickoff-async.md

---
title: Kickoff Async
description: Kickoff a Crew Asynchronously
---

## Introduction
CrewAI provides the ability to kickoff a crew asynchronously, allowing you to start the crew execution in a non-blocking manner. This feature is particularly useful when you want to run multiple crews concurrently or when you need to perform other tasks while the crew is executing.

## Asynchronous Crew Execution
To kickoff a crew asynchronously, use the `kickoff_async()` method. This method initiates the crew execution in a separate thread, allowing the main thread to continue executing other tasks.

Here's an example of how to kickoff a crew asynchronously:

```python
from crewai import Crew, Agent, Task

# Create an agent with code execution enabled
coding_agent = Agent(
    role="Python Data Analyst",
    goal="Analyze data and provide insights using Python",
    backstory="You are an experienced data analyst with strong Python skills.",
    allow_code_execution=True
)

# Create a task that requires code execution
data_analysis_task = Task(
    description="Analyze the given dataset and calculate the average age of participants. Ages: {ages}",
    agent=coding_agent
)

# Create a crew and add the task
analysis_crew = Crew(
    agents=[coding_agent],
    tasks=[data_analysis_task]
)

# Execute the crew
result = analysis_crew.kickoff_async(inputs={"ages": [25, 30, 35, 40, 45]})
```



---


# Kickoff-for-each.md

---
title: Kickoff For Each
description: Kickoff a Crew for a List
---

## Introduction
CrewAI provides the ability to kickoff a crew for each item in a list, allowing you to execute the crew for each item in the list. This feature is particularly useful when you need to perform the same set of tasks for multiple items.

## Kicking Off a Crew for Each Item
To kickoff a crew for each item in a list, use the `kickoff_for_each()` method. This method executes the crew for each item in the list, allowing you to process multiple items efficiently.

Here's an example of how to kickoff a crew for each item in a list:

```python
from crewai import Crew, Agent, Task

# Create an agent with code execution enabled
coding_agent = Agent(
    role="Python Data Analyst",
    goal="Analyze data and provide insights using Python",
    backstory="You are an experienced data analyst with strong Python skills.",
    allow_code_execution=True
)

# Create a task that requires code execution
data_analysis_task = Task(
    description="Analyze the given dataset and calculate the average age of participants. Ages: {ages}",
    agent=coding_agent
)

# Create a crew and add the task
analysis_crew = Crew(
    agents=[coding_agent],
    tasks=[data_analysis_task]
)

datasets = [
  { "ages": [25, 30, 35, 40, 45] },
  { "ages": [20, 25, 30, 35, 40] },
  { "ages": [30, 35, 40, 45, 50] }
]

# Execute the crew
result = analysis_crew.kickoff_for_each(inputs=datasets)
```


---


# LLM-Connections.md

---
title: Connect CrewAI to LLMs
description: Comprehensive guide on integrating CrewAI with various Large Language Models (LLMs), including detailed class attributes, methods, and configuration options.
---

## Connect CrewAI to LLMs

!!! note "Default LLM"
    By default, CrewAI uses OpenAI's GPT-4o model (specifically, the model specified by the OPENAI_MODEL_NAME environment variable, defaulting to "gpt-4o") for language processing. You can configure your agents to use a different model or API as described in this guide.
    By default, CrewAI uses OpenAI's GPT-4 model (specifically, the model specified by the OPENAI_MODEL_NAME environment variable, defaulting to "gpt-4") for language processing. You can configure your agents to use a different model or API as described in this guide.

CrewAI provides extensive versatility in integrating with various Language Models (LLMs), including local options through Ollama such as  Llama and Mixtral to cloud-based solutions like Azure. Its compatibility extends to all [LangChain LLM components](https://python.langchain.com/v0.2/docs/integrations/llms/), offering a wide range of integration possibilities for customized AI applications.

The platform supports connections to an array of Generative AI models, including:

 - OpenAI's suite of advanced language models
 - Anthropic's cutting-edge AI offerings
 - Ollama's diverse range of locally-hosted generative model & embeddings
 - LM Studio's diverse range of locally hosted generative models & embeddings
 - Groq's Super Fast LLM offerings
 - Azures' generative AI offerings
 - HuggingFace's generative AI offerings

This broad spectrum of LLM options enables users to select the most suitable model for their specific needs, whether prioritizing local deployment, specialized capabilities, or cloud-based scalability.

## Changing the default LLM
The default LLM is provided through the `langchain openai` package, which is installed by default when you install CrewAI. You can change this default LLM to a different model or API by setting the `OPENAI_MODEL_NAME` environment variable. This straightforward process allows you to harness the power of different OpenAI models, enhancing the flexibility and capabilities of your CrewAI implementation.
```python
# Required
os.environ["OPENAI_MODEL_NAME"]="gpt-4-0125-preview"

# Agent will automatically use the model defined in the environment variable
example_agent = Agent(
  role='Local Expert',
  goal='Provide insights about the city',
  backstory="A knowledgeable local guide.",
  verbose=True
)
```
## Ollama Local Integration
Ollama is preferred for local LLM integration, offering customization and privacy benefits. To integrate Ollama with CrewAI, you will need the `langchain-ollama` package. You can then set the following environment variables to connect to your Ollama instance running locally on port 11434.

```sh
os.environ[OPENAI_API_BASE]='http://localhost:11434'
os.environ[OPENAI_MODEL_NAME]='llama2'  # Adjust based on available model
os.environ[OPENAI_API_KEY]='' # No API Key required for Ollama
```

## Ollama Integration Step by Step (ex. for using Llama 3.1 8B locally)
1. [Download and install Ollama](https://ollama.com/download).   
2. After setting up the Ollama, Pull the Llama3.1 8B model by typing following lines into your terminal ```ollama run llama3.1```.   
3. Llama3.1 should now be served locally on `http://localhost:11434`
```
from crewai import Agent, Task, Crew
from langchain_ollama import ChatOllama
import os
os.environ["OPENAI_API_KEY"] = "NA"

llm = Ollama(
    model = "llama3.1",
    base_url = "http://localhost:11434")

general_agent = Agent(role = "Math Professor",
                      goal = """Provide the solution to the students that are asking mathematical questions and give them the answer.""",
                      backstory = """You are an excellent math professor that likes to solve math questions in a way that everyone can understand your solution""",
                      allow_delegation = False,
                      verbose = True,
                      llm = llm)

task = Task(description="""what is 3 + 5""",
             agent = general_agent,
             expected_output="A numerical answer.")

crew = Crew(
            agents=[general_agent],
            tasks=[task],
            verbose=2
        )

result = crew.kickoff()

print(result)
```

## HuggingFace Integration
There are a couple of different ways you can use HuggingFace to host your LLM.

### Your own HuggingFace endpoint
```python
from langchain_huggingface import HuggingFaceEndpoint,

llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

agent = Agent(
    role="HuggingFace Agent",
    goal="Generate text using HuggingFace",
    backstory="A diligent explorer of GitHub docs.",
    llm=llm
)
```

## OpenAI Compatible API Endpoints
Switch between APIs and models seamlessly using environment variables, supporting platforms like FastChat, LM Studio, Groq, and Mistral AI.

### Configuration Examples
#### FastChat
```sh
os.environ[OPENAI_API_BASE]="http://localhost:8001/v1"
os.environ[OPENAI_MODEL_NAME]='oh-2.5m7b-q51'
os.environ[OPENAI_API_KEY]=NA
```

#### LM Studio
Launch [LM Studio](https://lmstudio.ai) and go to the Server tab. Then select a model from the dropdown menu and wait for it to load. Once it's loaded, click the green Start Server button and use the URL, port, and API key that's shown (you can modify them). Below is an example of the default settings as of LM Studio 0.2.19:
```sh
os.environ[OPENAI_API_BASE]="http://localhost:1234/v1"
os.environ[OPENAI_API_KEY]="lm-studio"
```

#### Groq API
```sh
os.environ[OPENAI_API_KEY]=your-groq-api-key
os.environ[OPENAI_MODEL_NAME]='llama3-8b-8192'
os.environ[OPENAI_API_BASE]=https://api.groq.com/openai/v1
```

#### Mistral API
```sh
os.environ[OPENAI_API_KEY]=your-mistral-api-key
os.environ[OPENAI_API_BASE]=https://api.mistral.ai/v1
os.environ[OPENAI_MODEL_NAME]="mistral-small"
```

### Solar
```sh
from langchain_community.chat_models.solar import SolarChat
```
```sh
os.environ[SOLAR_API_BASE]="https://api.upstage.ai/v1/solar"
os.environ[SOLAR_API_KEY]="your-solar-api-key"
```

# Free developer API key available here: https://console.upstage.ai/services/solar
# Langchain Example: https://github.com/langchain-ai/langchain/pull/18556


### Cohere
```python
from langchain_cohere import ChatCohere
# Initialize language model
os.environ["COHERE_API_KEY"] = "your-cohere-api-key"
llm = ChatCohere()

# Free developer API key available here: https://cohere.com/
# Langchain Documentation: https://python.langchain.com/docs/integrations/chat/cohere
```

### Azure Open AI Configuration
For Azure OpenAI API integration, set the following environment variables:
```sh

os.environ[AZURE_OPENAI_DEPLOYMENT] = "You deployment"
os.environ["OPENAI_API_VERSION"] = "2023-12-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = "Your Endpoint"
os.environ["AZURE_OPENAI_API_KEY"] = "<Your API Key>"
```

### Example Agent with Azure LLM
```python
from dotenv import load_dotenv
from crewai import Agent
from langchain_openai import AzureChatOpenAI

load_dotenv()

azure_llm = AzureChatOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_KEY")
)

azure_agent = Agent(
  role='Example Agent',
  goal='Demonstrate custom LLM configuration',
  backstory='A diligent explorer of GitHub docs.',
  llm=azure_llm
)
```
## Conclusion
Integrating CrewAI with different LLMs expands the framework's versatility, allowing for customized, efficient AI solutions across various domains and platforms.


---


# Langtrace-Observability.md

---
title: CrewAI Agent Monitoring with Langtrace
description: How to monitor cost, latency, and performance of CrewAI Agents using Langtrace, an external observability tool.
---

# Langtrace Overview

Langtrace is an open-source, external tool that helps you set up observability and evaluations for Large Language Models (LLMs), LLM frameworks, and Vector Databases. While not built directly into CrewAI, Langtrace can be used alongside CrewAI to gain deep visibility into the cost, latency, and performance of your CrewAI Agents. This integration allows you to log hyperparameters, monitor performance regressions, and establish a process for continuous improvement of your Agents.

## Setup Instructions

1. Sign up for [Langtrace](https://langtrace.ai/) by visiting [https://langtrace.ai/signup](https://langtrace.ai/signup).
2. Create a project and generate an API key.
3. Install Langtrace in your CrewAI project using the following commands:

```bash
# Install the SDK
pip install langtrace-python-sdk
```

## Using Langtrace with CrewAI

To integrate Langtrace with your CrewAI project, follow these steps:

1. Import and initialize Langtrace at the beginning of your script, before any CrewAI imports:

```python
from langtrace_python_sdk import langtrace
langtrace.init(api_key='<LANGTRACE_API_KEY>')

# Now import CrewAI modules
from crewai import Agent, Task, Crew
```

2. Create your CrewAI agents and tasks as usual.

3. Use Langtrace's tracking functions to monitor your CrewAI operations. For example:

```python
with langtrace.trace("CrewAI Task Execution"):
    result = crew.kickoff()
```

### Features and Their Application to CrewAI

1. **LLM Token and Cost Tracking**
   - Monitor the token usage and associated costs for each CrewAI agent interaction.
   - Example:
     ```python
     with langtrace.trace("Agent Interaction"):
         agent_response = agent.execute(task)
     ```

2. **Trace Graph for Execution Steps**
   - Visualize the execution flow of your CrewAI tasks, including latency and logs.
   - Useful for identifying bottlenecks in your agent workflows.

3. **Dataset Curation with Manual Annotation**
   - Create datasets from your CrewAI task outputs for future training or evaluation.
   - Example:
     ```python
     langtrace.log_dataset_item(task_input, agent_output, {"task_type": "research"})
     ```

4. **Prompt Versioning and Management**
   - Keep track of different versions of prompts used in your CrewAI agents.
   - Useful for A/B testing and optimizing agent performance.

5. **Prompt Playground with Model Comparisons**
   - Test and compare different prompts and models for your CrewAI agents before deployment.

6. **Testing and Evaluations**
   - Set up automated tests for your CrewAI agents and tasks.
   - Example:
     ```python
     langtrace.evaluate(agent_output, expected_output, "accuracy")
     ```

## Monitoring New CrewAI Features

CrewAI has introduced several new features that can be monitored using Langtrace:

1. **Code Execution**: Monitor the performance and output of code executed by agents.
   ```python
   with langtrace.trace("Agent Code Execution"):
       code_output = agent.execute_code(code_snippet)
   ```

2. **Third-party Agent Integration**: Track interactions with LlamaIndex, LangChain, and Autogen agents.

---


# Replay-tasks-from-latest-Crew-Kickoff.md

---
title: Replay Tasks from Latest Crew Kickoff
description: Replay tasks from the latest crew.kickoff(...)
---

## Introduction
CrewAI provides the ability to replay from a task specified from the latest crew kickoff. This feature is particularly useful when you've finished a kickoff and may want to retry certain tasks or don't need to refetch data over and your agents already have the context saved from the kickoff execution so you just need to replay the tasks you want to.

## Note:
You must run `crew.kickoff()` before you can replay a task. Currently, only the latest kickoff is supported, so if you use `kickoff_for_each`, it will only allow you to replay from the most recent crew run.

Here's an example of how to replay from a task:

### Replaying from specific task Using the CLI
To use the replay feature, follow these steps:

1. Open your terminal or command prompt.
2. Navigate to the directory where your CrewAI project is located.
3. Run the following command:

To view latest kickoff task_ids use:
```shell
crewai log-tasks-outputs
```

Once you have your task_id to replay from use:
```shell
crewai replay -t <task_id>
```


### Replaying from a task Programmatically
To replay from a task programmatically, use the following steps:

1. Specify the task_id and input parameters for the replay process.
2. Execute the replay command within a try-except block to handle potential errors.

```python
   def replay():
    """
    Replay the crew execution from a specific task.
    """
    task_id = '<task_id>'
    inputs = {"topic": "CrewAI Training"} # this is optional, you can pass in the inputs you want to replay otherwise uses the previous kickoffs inputs
    try:
        YourCrewName_Crew().crew().replay(task_id=task_id, inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

---


# Sequential.md

---
title: Using the Sequential Processes in crewAI
description: A comprehensive guide to utilizing the sequential processes for task execution in crewAI projects.
---

## Introduction
CrewAI offers a flexible framework for executing tasks in a structured manner, supporting both sequential and hierarchical processes. This guide outlines how to effectively implement these processes to ensure efficient task execution and project completion.

## Sequential Process Overview
The sequential process ensures tasks are executed one after the other, following a linear progression. This approach is ideal for projects requiring tasks to be completed in a specific order.

### Key Features
- **Linear Task Flow**: Ensures orderly progression by handling tasks in a predetermined sequence.
- **Simplicity**: Best suited for projects with clear, step-by-step tasks.
- **Easy Monitoring**: Facilitates easy tracking of task completion and project progress.

## Implementing the Sequential Process
To use the sequential process, assemble your crew and define tasks in the order they need to be executed.

```python
from crewai import Crew, Process, Agent, Task

# Define your agents
researcher = Agent(
  role='Researcher',
  goal='Conduct foundational research',
  backstory='An experienced researcher with a passion for uncovering insights'
)
analyst = Agent(
  role='Data Analyst',
  goal='Analyze research findings',
  backstory='A meticulous analyst with a knack for uncovering patterns'
)
writer = Agent(
  role='Writer',
  goal='Draft the final report',
  backstory='A skilled writer with a talent for crafting compelling narratives'
)

research_task = Task(description='Gather relevant data...', agent=researcher, expected_output='Raw Data')
analysis_task = Task(description='Analyze the data...', agent=analyst, expected_output='Data Insights')
writing_task = Task(description='Compose the report...', agent=writer, expected_output='Final Report')

# Form the crew with a sequential process
report_crew = Crew(
  agents=[researcher, analyst, writer],
  tasks=[research_task, analysis_task, writing_task],
  process=Process.sequential
)

# Execute the crew
result = report_crew.kickoff()
```

### Workflow in Action
1. **Initial Task**: In a sequential process, the first agent completes their task and signals completion.
2. **Subsequent Tasks**: Agents pick up their tasks based on the process type, with outcomes of preceding tasks or manager directives guiding their execution.
3. **Completion**: The process concludes once the final task is executed, leading to project completion.

## Advanced Features

### Task Delegation
In sequential processes, if an agent has `allow_delegation` set to `True`, they can delegate tasks to other agents in the crew. This feature is automatically set up when there are multiple agents in the crew.

### Asynchronous Execution
Tasks can be executed asynchronously, allowing for parallel processing when appropriate. To create an asynchronous task, set `async_execution=True` when defining the task.

### Memory and Caching
CrewAI supports both memory and caching features:
- **Memory**: Enable by setting `memory=True` when creating the Crew. This allows agents to retain information across tasks.
- **Caching**: By default, caching is enabled. Set `cache=False` to disable it.

### Callbacks
You can set callbacks at both the task and step level:
- `task_callback`: Executed after each task completion.
- `step_callback`: Executed after each step in an agent's execution.

### Usage Metrics
CrewAI tracks token usage across all tasks and agents. You can access these metrics after execution.

## Best Practices for Sequential Processes
1. **Order Matters**: Arrange tasks in a logical sequence where each task builds upon the previous one.
2. **Clear Task Descriptions**: Provide detailed descriptions for each task to guide the agents effectively.
3. **Appropriate Agent Selection**: Match agents' skills and roles to the requirements of each task.
4. **Use Context**: Leverage the context from previous tasks to inform subsequent ones


---


# Your-Own-Manager-Agent.md

---
title: Setting a Specific Agent as Manager in CrewAI
description: Learn how to set a custom agent as the manager in CrewAI, providing more control over task management and coordination.

---

# Setting a Specific Agent as Manager in CrewAI

CrewAI allows users to set a specific agent as the manager of the crew, providing more control over the management and coordination of tasks. This feature enables the customization of the managerial role to better fit your project's requirements.

## Using the `manager_agent` Attribute

### Custom Manager Agent

The `manager_agent` attribute allows you to define a custom agent to manage the crew. This agent will oversee the entire process, ensuring that tasks are completed efficiently and to the highest standard.

### Example

```python
import os
from crewai import Agent, Task, Crew, Process

# Define your agents
researcher = Agent(
    role="Researcher",
    goal="Conduct thorough research and analysis on AI and AI agents",
    backstory="You're an expert researcher, specialized in technology, software engineering, AI, and startups. You work as a freelancer and are currently researching for a new client.",
    allow_delegation=False,
)

writer = Agent(
    role="Senior Writer",
    goal="Create compelling content about AI and AI agents",
    backstory="You're a senior writer, specialized in technology, software engineering, AI, and startups. You work as a freelancer and are currently writing content for a new client.",
    allow_delegation=False,
)

# Define your task
task = Task(
    description="Generate a list of 5 interesting ideas for an article, then write one captivating paragraph for each idea that showcases the potential of a full article on this topic. Return the list of ideas with their paragraphs and your notes.",
    expected_output="5 bullet points, each with a paragraph and accompanying notes.",
)

# Define the manager agent
manager = Agent(
    role="Project Manager",
    goal="Efficiently manage the crew and ensure high-quality task completion",
    backstory="You're an experienced project manager, skilled in overseeing complex projects and guiding teams to success. Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on time and to the highest standard.",
    allow_delegation=True,
)

# Instantiate your crew with a custom manager
crew = Crew(
    agents=[researcher, writer],
    tasks=[task],
    manager_agent=manager,
    process=Process.hierarchical,
)

# Start the crew's work
result = crew.kickoff()
```

## Benefits of a Custom Manager Agent

- **Enhanced Control**: Tailor the management approach to fit the specific needs of your project.
- **Improved Coordination**: Ensure efficient task coordination and management by an experienced agent.
- **Customizable Management**: Define managerial roles and responsibilities that align with your project's goals.

## Setting a Manager LLM

If you're using the hierarchical process and don't want to set a custom manager agent, you can specify the language model for the manager:

```python
from langchain_openai import ChatOpenAI

manager_llm = ChatOpenAI(model_name="gpt-4")

crew = Crew(
    agents=[researcher, writer],
    tasks=[task],
    process=Process.hierarchical,
    manager_llm=manager_llm
)
```

Note: Either `manager_agent` or `manager_llm` must be set when using the hierarchical process.

---


# BrowserbaseLoadTool.md

# BrowserbaseLoadTool

## Description

[Browserbase](https://browserbase.com) is a developer platform to reliably run, manage, and monitor headless browsers.

 Power your AI data retrievals with:
 - [Serverless Infrastructure](https://docs.browserbase.com/under-the-hood) providing reliable browsers to extract data from complex UIs
 - [Stealth Mode](https://docs.browserbase.com/features/stealth-mode) with included fingerprinting tactics and automatic captcha solving
 - [Session Debugger](https://docs.browserbase.com/features/sessions) to inspect your Browser Session with networks timeline and logs
 - [Live Debug](https://docs.browserbase.com/guides/session-debug-connection/browser-remote-control) to quickly debug your automation

## Installation

- Get an API key and Project ID from [browserbase.com](https://browserbase.com) and set it in environment variables (`BROWSERBASE_API_KEY`, `BROWSERBASE_PROJECT_ID`).
- Install the [Browserbase SDK](http://github.com/browserbase/python-sdk) along with `crewai[tools]` package:

```
pip install browserbase 'crewai[tools]'
```

## Example

Utilize the BrowserbaseLoadTool as follows to allow your agent to load websites:

```python
from crewai_tools import BrowserbaseLoadTool

tool = BrowserbaseLoadTool()
```

## Arguments

- `api_key` Optional. Browserbase API key. Default is `BROWSERBASE_API_KEY` env variable.
- `project_id` Optional. Browserbase Project ID. Default is `BROWSERBASE_PROJECT_ID` env variable.
- `text_content` Retrieve only text content. Default is `False`.
- `session_id` Optional. Provide an existing Session ID.
- `proxy` Optional. Enable/Disable Proxies."


---


# CSVSearchTool.md

# CSVSearchTool

!!! note "Experimental"
    We are still working on improving tools, so there might be unexpected behavior or changes in the future.

## Description

This tool is used to perform a RAG (Retrieval-Augmented Generation) search within a CSV file's content. It allows users to semantically search for queries in the content of a specified CSV file. This feature is particularly useful for extracting information from large CSV datasets where traditional search methods might be inefficient. All tools with "Search" in their name, including CSVSearchTool, are RAG tools designed for searching different sources of data.

## Installation

Install the crewai_tools package

```shell
pip install 'crewai[tools]'
```

## Example

```python
from crewai_tools import CSVSearchTool

# Initialize the tool with a specific CSV file. This setup allows the agent to only search the given CSV file.
tool = CSVSearchTool(csv='path/to/your/csvfile.csv')

# OR

# Initialize the tool without a specific CSV file. Agent  will need to provide the CSV path at runtime.
tool = CSVSearchTool()
```

## Arguments

- `csv` : The path to the CSV file you want to search. This is a mandatory argument if the tool was initialized without a specific CSV file; otherwise, it is optional.

## Custom model and embeddings

By default, the tool uses OpenAI for both embeddings and summarization. To customize the model, you can use a config dictionary as follows:

```python
tool = CSVSearchTool(
    config=dict(
        llm=dict(
            provider="ollama", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="llama2",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="google", # or openai, ollama, ...
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)
```


---


# CodeDocsSearchTool.md

# CodeDocsSearchTool

!!! note "Experimental"
    We are still working on improving tools, so there might be unexpected behavior or changes in the future.

## Description

The CodeDocsSearchTool is a powerful RAG (Retrieval-Augmented Generation) tool designed for semantic searches within code documentation. It enables users to efficiently find specific information or topics within code documentation. By providing a `docs_url` during initialization, the tool narrows down the search to that particular documentation site. Alternatively, without a specific `docs_url`, it searches across a wide array of code documentation known or discovered throughout its execution, making it versatile for various documentation search needs.

## Installation

To start using the CodeDocsSearchTool, first, install the crewai_tools package via pip:

```
pip install 'crewai[tools]'
```

## Example

Utilize the CodeDocsSearchTool as follows to conduct searches within code documentation:

```python
from crewai_tools import CodeDocsSearchTool

# To search any code documentation content if the URL is known or discovered during its execution:
tool = CodeDocsSearchTool()

# OR

# To specifically focus your search on a given documentation site by providing its URL:
tool = CodeDocsSearchTool(docs_url='https://docs.example.com/reference')
```
Note: Substitute 'https://docs.example.com/reference' with your target documentation URL and 'How to use search tool' with the search query relevant to your needs.

## Arguments

- `docs_url`: Optional. Specifies the URL of the code documentation to be searched. Providing this during the tool's initialization focuses the search on the specified documentation content.

## Custom model and embeddings

By default, the tool uses OpenAI for both embeddings and summarization. To customize the model, you can use a config dictionary as follows:

```python
tool = CodeDocsSearchTool(
    config=dict(
        llm=dict(
            provider="ollama", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="llama2",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="google", # or openai, ollama, ...
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)
```


---


# CodeInterpreterTool.md

# CodeInterpreterTool

## Description
This tool enables the Agent to execute Python 3 code that it has generated autonomously. The code is run in a secure, isolated environment, ensuring safety regardless of the content. 

This functionality is particularly valuable as it allows the Agent to create code, execute it within the same ecosystem, obtain the results, and utilize that information to inform subsequent decisions and actions.

## Requirements

- Docker

## Installation
Install the crewai_tools package
```shell
pip install 'crewai[tools]'
```

## Example

Remember that when using this tool, the code must be generated by the Agent itself. The code must be a Python3 code. And it will take some time for the first time to run because it needs to build the Docker image.

```python
from crewai import Agent
from crewai_tools import CodeInterpreterTool

Agent(
    ...
    tools=[CodeInterpreterTool()],
)
```

We also provide a simple way to use it directly from the Agent.

```python
from crewai import Agent

agent = Agent(
    ...
    allow_code_execution=True,
)
```


---


# ComposioTool.md

# ComposioTool Documentation

## Description

This tools is a wrapper around the composio set of tools and gives your agent access to a wide variety of tools from the composio SDK.

## Installation

To incorporate this tool into your project, follow the installation instructions below:

```shell
pip install composio-core
pip install 'crewai[tools]'
```

after the installation is complete, either run `composio login` or export your composio API key as `COMPOSIO_API_KEY`.

## Example

The following example demonstrates how to initialize the tool and execute a github action:

1. Initialize Composio tools

```python
from composio import App
from crewai_tools import ComposioTool
from crewai import Agent, Task


tools = [ComposioTool.from_action(action=Action.GITHUB_ACTIVITY_STAR_REPO_FOR_AUTHENTICATED_USER)]
```

If you don't know what action you want to use, use `from_app` and `tags` filter to get relevant actions

```python
tools = ComposioTool.from_app(App.GITHUB, tags=["important"])
```

or use `use_case` to search relevant actions

```python
tools = ComposioTool.from_app(App.GITHUB, use_case="Star a github repository")
```

2. Define agent

```python
crewai_agent = Agent(
    role="Github Agent",
    goal="You take action on Github using Github APIs",
    backstory=(
        "You are AI agent that is responsible for taking actions on Github "
        "on users behalf. You need to take action on Github using Github APIs"
    ),
    verbose=True,
    tools=tools,
)
```

3. Execute task

```python
task = Task(
    description="Star a repo ComposioHQ/composio on GitHub",
    agent=crewai_agent,
    expected_output="if the star happened",
)

task.execute()
```

* More detailed list of tools can be found [here](https://app.composio.dev)


---


# DOCXSearchTool.md

# DOCXSearchTool

!!! note "Experimental"
    We are still working on improving tools, so there might be unexpected behavior or changes in the future.

## Description
The DOCXSearchTool is a RAG tool designed for semantic searching within DOCX documents. It enables users to effectively search and extract relevant information from DOCX files using query-based searches. This tool is invaluable for data analysis, information management, and research tasks, streamlining the process of finding specific information within large document collections.

## Installation
Install the crewai_tools package by running the following command in your terminal:

```shell
pip install 'crewai[tools]'
```

## Example
The following example demonstrates initializing the DOCXSearchTool to search within any DOCX file's content or with a specific DOCX file path.

```python
from crewai_tools import DOCXSearchTool

# Initialize the tool to search within any DOCX file's content
tool = DOCXSearchTool()

# OR

# Initialize the tool with a specific DOCX file, so the agent can only search the content of the specified DOCX file
tool = DOCXSearchTool(docx='path/to/your/document.docx')
```

## Arguments
- `docx`: An optional file path to a specific DOCX document you wish to search. If not provided during initialization, the tool allows for later specification of any DOCX file's content path for searching.

## Custom model and embeddings

By default, the tool uses OpenAI for both embeddings and summarization. To customize the model, you can use a config dictionary as follows:

```python
tool = DOCXSearchTool(
    config=dict(
        llm=dict(
            provider="ollama", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="llama2",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="google", # or openai, ollama, ...
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)
```


---


# DirectoryReadTool.md

```markdown
# DirectoryReadTool

!!! note "Experimental"
    We are still working on improving tools, so there might be unexpected behavior or changes in the future.

## Description
The DirectoryReadTool is a powerful utility designed to provide a comprehensive listing of directory contents. It can recursively navigate through the specified directory, offering users a detailed enumeration of all files, including those within subdirectories. This tool is crucial for tasks that require a thorough inventory of directory structures or for validating the organization of files within directories.

## Installation
To utilize the DirectoryReadTool in your project, install the `crewai_tools` package. If this package is not yet part of your environment, you can install it using pip with the command below:

```shell
pip install 'crewai[tools]'
```

This command installs the latest version of the `crewai_tools` package, granting access to the DirectoryReadTool among other utilities.

## Example
Employing the DirectoryReadTool is straightforward. The following code snippet demonstrates how to set it up and use the tool to list the contents of a specified directory:

```python
from crewai_tools import DirectoryReadTool

# Initialize the tool so the agent can read any directory's content it learns about during execution
tool = DirectoryReadTool()

# OR

# Initialize the tool with a specific directory, so the agent can only read the content of the specified directory
tool = DirectoryReadTool(directory='/path/to/your/directory')
```

## Arguments
The DirectoryReadTool requires minimal configuration for use. The essential argument for this tool is as follows:

- `directory`: **Optional**. An argument that specifies the path to the directory whose contents you wish to list. It accepts both absolute and relative paths, guiding the tool to the desired directory for content listing.

---


# DirectorySearchTool.md

# DirectorySearchTool

!!! note "Experimental"
    The DirectorySearchTool is under continuous development. Features and functionalities might evolve, and unexpected behavior may occur as we refine the tool.

## Description
The DirectorySearchTool enables semantic search within the content of specified directories, leveraging the Retrieval-Augmented Generation (RAG) methodology for efficient navigation through files. Designed for flexibility, it allows users to dynamically specify search directories at runtime or set a fixed directory during initial setup.

## Installation
To use the DirectorySearchTool, begin by installing the crewai_tools package. Execute the following command in your terminal:

```shell
pip install 'crewai[tools]'
```

## Initialization and Usage
Import the DirectorySearchTool from the `crewai_tools` package to start. You can initialize the tool without specifying a directory, enabling the setting of the search directory at runtime. Alternatively, the tool can be initialized with a predefined directory.

```python
from crewai_tools import DirectorySearchTool

# For dynamic directory specification at runtime
tool = DirectorySearchTool()

# For fixed directory searches
tool = DirectorySearchTool(directory='/path/to/directory')
```

## Arguments
- `directory`: A string argument that specifies the search directory. This is optional during initialization but required for searches if not set initially.

## Custom Model and Embeddings
The DirectorySearchTool uses OpenAI for embeddings and summarization by default. Customization options for these settings include changing the model provider and configuration, enhancing flexibility for advanced users.

```python
tool = DirectorySearchTool(
    config=dict(
        llm=dict(
            provider="ollama", # Options include ollama, google, anthropic, llama2, and more
            config=dict(
                model="llama2",
                # Additional configurations here
            ),
        ),
        embedder=dict(
            provider="google", # or openai, ollama, ...
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)
```

---


# EXASearchTool.md

# EXASearchTool Documentation

## Description

The EXASearchTool is designed to perform a semantic search for a specified query from a text's content across the internet. It utilizes the [exa.ai](https://exa.ai/) API to fetch and display the most relevant search results based on the query provided by the user.

## Installation

To incorporate this tool into your project, follow the installation instructions below:

```shell
pip install 'crewai[tools]'
```

## Example

The following example demonstrates how to initialize the tool and execute a search with a given query:

```python
from crewai_tools import EXASearchTool

# Initialize the tool for internet searching capabilities
tool = EXASearchTool()
```

## Steps to Get Started

To effectively use the EXASearchTool, follow these steps:

1. **Package Installation**: Confirm that the `crewai[tools]` package is installed in your Python environment.
2. **API Key Acquisition**: Acquire a [exa.ai](https://exa.ai/) API key by registering for a free account at [exa.ai](https://exa.ai/).
3. **Environment Configuration**: Store your obtained API key in an environment variable named `EXA_API_KEY` to facilitate its use by the tool.

## Conclusion

By integrating the EXASearchTool into Python projects, users gain the ability to conduct real-time, relevant searches across the internet directly from their applications. By adhering to the setup and usage guidelines provided, incorporating this tool into projects is streamlined and straightforward.


---


# FileReadTool.md

# FileReadTool

!!! note "Experimental"
    We are still working on improving tools, so there might be unexpected behavior or changes in the future.

## Description
The FileReadTool conceptually represents a suite of functionalities within the crewai_tools package aimed at facilitating file reading and content retrieval. This suite includes tools for processing batch text files, reading runtime configuration files, and importing data for analytics. It supports a variety of text-based file formats such as `.txt`, `.csv`, `.json`, and more. Depending on the file type, the suite offers specialized functionality, such as converting JSON content into a Python dictionary for ease of use.

## Installation
To utilize the functionalities previously attributed to the FileReadTool, install the crewai_tools package:

```shell
pip install 'crewai[tools]'
```

## Usage Example
To get started with the FileReadTool:

```python
from crewai_tools import FileReadTool

# Initialize the tool to read any files the agents knows or lean the path for
file_read_tool = FileReadTool()

# OR

# Initialize the tool with a specific file path, so the agent can only read the content of the specified file
file_read_tool = FileReadTool(file_path='path/to/your/file.txt')
```

## Arguments
- `file_path`: The path to the file you want to read. It accepts both absolute and relative paths. Ensure the file exists and you have the necessary permissions to access it.

---


# GitHubSearchTool.md

# GithubSearchTool

!!! note "Experimental"
    We are still working on improving tools, so there might be unexpected behavior or changes in the future.

## Description
The GithubSearchTool is a Retrieval-Augmented Generation (RAG) tool specifically designed for conducting semantic searches within GitHub repositories. Utilizing advanced semantic search capabilities, it sifts through code, pull requests, issues, and repositories, making it an essential tool for developers, researchers, or anyone in need of precise information from GitHub.

## Installation
To use the GithubSearchTool, first ensure the crewai_tools package is installed in your Python environment:

```shell
pip install 'crewai[tools]'
```

This command installs the necessary package to run the GithubSearchTool along with any other tools included in the crewai_tools package.

## Example
Here’s how you can use the GithubSearchTool to perform semantic searches within a GitHub repository:
```python
from crewai_tools import GithubSearchTool

# Initialize the tool for semantic searches within a specific GitHub repository
tool = GithubSearchTool(
	github_repo='https://github.com/example/repo',
	content_types=['code', 'issue'] # Options: code, repo, pr, issue
)

# OR

# Initialize the tool for semantic searches within a specific GitHub repository, so the agent can search any repository if it learns about during its execution
tool = GithubSearchTool(
	content_types=['code', 'issue'] # Options: code, repo, pr, issue
)
```

## Arguments
- `github_repo` : The URL of the GitHub repository where the search will be conducted. This is a mandatory field and specifies the target repository for your search.
- `content_types` : Specifies the types of content to include in your search. You must provide a list of content types from the following options: `code` for searching within the code, `repo` for searching within the repository's general information, `pr` for searching within pull requests, and `issue` for searching within issues. This field is mandatory and allows tailoring the search to specific content types within the GitHub repository.

## Custom model and embeddings

By default, the tool uses OpenAI for both embeddings and summarization. To customize the model, you can use a config dictionary as follows:

```python
tool = GithubSearchTool(
    config=dict(
        llm=dict(
            provider="ollama", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="llama2",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="google", # or openai, ollama, ...
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)
```


---


# JSONSearchTool.md

# JSONSearchTool

!!! note "Experimental Status"
    The JSONSearchTool is currently in an experimental phase. This means the tool is under active development, and users might encounter unexpected behavior or changes. We highly encourage feedback on any issues or suggestions for improvements.

## Description
The JSONSearchTool is designed to facilitate efficient and precise searches within JSON file contents. It utilizes a RAG (Retrieve and Generate) search mechanism, allowing users to specify a JSON path for targeted searches within a particular JSON file. This capability significantly improves the accuracy and relevance of search results.

## Installation
To install the JSONSearchTool, use the following pip command:

```shell
pip install 'crewai[tools]'
```

## Usage Examples
Here are updated examples on how to utilize the JSONSearchTool effectively for searching within JSON files. These examples take into account the current implementation and usage patterns identified in the codebase.

```python
from crewai.json_tools import JSONSearchTool  # Updated import path

# General JSON content search
# This approach is suitable when the JSON path is either known beforehand or can be dynamically identified.
tool = JSONSearchTool()

# Restricting search to a specific JSON file
# Use this initialization method when you want to limit the search scope to a specific JSON file.
tool = JSONSearchTool(json_path='./path/to/your/file.json')
```

## Arguments
- `json_path` (str, optional): Specifies the path to the JSON file to be searched. This argument is not required if the tool is initialized for a general search. When provided, it confines the search to the specified JSON file.

## Configuration Options
The JSONSearchTool supports extensive customization through a configuration dictionary. This allows users to select different models for embeddings and summarization based on their requirements.

```python
tool = JSONSearchTool(
    config={
        "llm": {
            "provider": "ollama",  # Other options include google, openai, anthropic, llama2, etc.
            "config": {
                "model": "llama2",
                # Additional optional configurations can be specified here.
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            },
        },
        "embedder": {
            "provider": "google", # or openai, ollama, ...
            "config": {
                "model": "models/embedding-001",
                "task_type": "retrieval_document",
                # Further customization options can be added here.
            },
        },
    }
)
```

---


# MDXSearchTool.md

# MDXSearchTool

!!! note "Experimental"
    The MDXSearchTool is in continuous development. Features may be added or removed, and functionality could change unpredictably as we refine the tool.

## Description
The MDX Search Tool is a component of the `crewai_tools` package aimed at facilitating advanced markdown language extraction. It enables users to effectively search and extract relevant information from MD files using query-based searches. This tool is invaluable for data analysis, information management, and research tasks, streamlining the process of finding specific information within large document collections.

## Installation
Before using the MDX Search Tool, ensure the `crewai_tools` package is installed. If it is not, you can install it with the following command:

```shell
pip install 'crewai[tools]'
```

## Usage Example
To use the MDX Search Tool, you must first set up the necessary environment variables. Then, integrate the tool into your crewAI project to begin your market research. Below is a basic example of how to do this:

```python
from crewai_tools import MDXSearchTool

# Initialize the tool to search any MDX content it learns about during execution
tool = MDXSearchTool()

# OR

# Initialize the tool with a specific MDX file path for an exclusive search within that document
tool = MDXSearchTool(mdx='path/to/your/document.mdx')
```

## Parameters
- mdx: **Optional**. Specifies the MDX file path for the search. It can be provided during initialization.

## Customization of Model and Embeddings

The tool defaults to using OpenAI for embeddings and summarization. For customization, utilize a configuration dictionary as shown below:

```python
tool = MDXSearchTool(
    config=dict(
        llm=dict(
            provider="ollama", # Options include google, openai, anthropic, llama2, etc.
            config=dict(
                model="llama2",
                # Optional parameters can be included here.
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="google", # or openai, ollama, ...
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
                # Optional title for the embeddings can be added here.
                # title="Embeddings",
            ),
        ),
    )
)
```


---


# PDFSearchTool.md

# PDFSearchTool

!!! note "Experimental"
    We are still working on improving tools, so there might be unexpected behavior or changes in the future.

## Description
The PDFSearchTool is a RAG tool designed for semantic searches within PDF content. It allows for inputting a search query and a PDF document, leveraging advanced search techniques to find relevant content efficiently. This capability makes it especially useful for extracting specific information from large PDF files quickly.

## Installation
To get started with the PDFSearchTool, first, ensure the crewai_tools package is installed with the following command:

```shell
pip install 'crewai[tools]'
```

## Example
Here's how to use the PDFSearchTool to search within a PDF document:

```python
from crewai_tools import PDFSearchTool

# Initialize the tool allowing for any PDF content search if the path is provided during execution
tool = PDFSearchTool()

# OR

# Initialize the tool with a specific PDF path for exclusive search within that document
tool = PDFSearchTool(pdf='path/to/your/document.pdf')
```

## Arguments
- `pdf`: **Optional** The PDF path for the search. Can be provided at initialization or within the `run` method's arguments. If provided at initialization, the tool confines its search to the specified document.

## Custom model and embeddings

By default, the tool uses OpenAI for both embeddings and summarization. To customize the model, you can use a config dictionary as follows:

```python
tool = PDFSearchTool(
    config=dict(
        llm=dict(
            provider="ollama", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="llama2",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="google", # or openai, ollama, ...
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)
```


---


# PGSearchTool.md

# PGSearchTool

!!! note "Under Development"
    The PGSearchTool is currently under development. This document outlines the intended functionality and interface. As development progresses, please be aware that some features may not be available or could change.

## Description
The PGSearchTool is envisioned as a powerful tool for facilitating semantic searches within PostgreSQL database tables. By leveraging advanced Retrieve and Generate (RAG) technology, it aims to provide an efficient means for querying database table content, specifically tailored for PostgreSQL databases. The tool's goal is to simplify the process of finding relevant data through semantic search queries, offering a valuable resource for users needing to conduct advanced queries on extensive datasets within a PostgreSQL environment.

## Installation
The `crewai_tools` package, which will include the PGSearchTool upon its release, can be installed using the following command:

```shell
pip install 'crewai[tools]'
```

(Note: The PGSearchTool is not yet available in the current version of the `crewai_tools` package. This installation command will be updated once the tool is released.)

## Example Usage
Below is a proposed example showcasing how to use the PGSearchTool for conducting a semantic search on a table within a PostgreSQL database:

```python
from crewai_tools import PGSearchTool

# Initialize the tool with the database URI and the target table name
tool = PGSearchTool(db_uri='postgresql://user:password@localhost:5432/mydatabase', table_name='employees')
```

## Arguments
The PGSearchTool is designed to require the following arguments for its operation:

- `db_uri`: A string representing the URI of the PostgreSQL database to be queried. This argument will be mandatory and must include the necessary authentication details and the location of the database.
- `table_name`: A string specifying the name of the table within the database on which the semantic search will be performed. This argument will also be mandatory.

## Custom Model and Embeddings

The tool intends to use OpenAI for both embeddings and summarization by default. Users will have the option to customize the model using a config dictionary as follows:

```python
tool = PGSearchTool(
    config=dict(
        llm=dict(
            provider="ollama", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="llama2",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="google", # or openai, ollama, ...
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)
```


---


# ScrapeWebsiteTool.md

# ScrapeWebsiteTool

!!! note "Experimental"
    We are still working on improving tools, so there might be unexpected behavior or changes in the future.

## Description
A tool designed to extract and read the content of a specified website. It is capable of handling various types of web pages by making HTTP requests and parsing the received HTML content. This tool can be particularly useful for web scraping tasks, data collection, or extracting specific information from websites.

## Installation
Install the crewai_tools package
```shell
pip install 'crewai[tools]'
```

## Example
```python
from crewai_tools import ScrapeWebsiteTool

# To enable scrapping any website it finds during it's execution
tool = ScrapeWebsiteTool()

# Initialize the tool with the website URL, so the agent can only scrap the content of the specified website
tool = ScrapeWebsiteTool(website_url='https://www.example.com')

# Extract the text from the site
text = tool.run()
print(text)
```

## Arguments
- `website_url` : Mandatory website URL to read the file. This is the primary input for the tool, specifying which website's content should be scraped and read.


---


# SeleniumScrapingTool.md

# SeleniumScrapingTool

!!! note "Experimental"
    This tool is currently in development. As we refine its capabilities, users may encounter unexpected behavior. Your feedback is invaluable to us for making improvements.

## Description
The SeleniumScrapingTool is crafted for high-efficiency web scraping tasks. It allows for precise extraction of content from web pages by using CSS selectors to target specific elements. Its design caters to a wide range of scraping needs, offering flexibility to work with any provided website URL.

## Installation
To get started with the SeleniumScrapingTool, install the crewai_tools package using pip:

```
pip install 'crewai[tools]'
```

## Usage Examples
Below are some scenarios where the SeleniumScrapingTool can be utilized:

```python
from crewai_tools import SeleniumScrapingTool

# Example 1: Initialize the tool without any parameters to scrape the current page it navigates to
tool = SeleniumScrapingTool()

# Example 2: Scrape the entire webpage of a given URL
tool = SeleniumScrapingTool(website_url='https://example.com')

# Example 3: Target and scrape a specific CSS element from a webpage
tool = SeleniumScrapingTool(website_url='https://example.com', css_element='.main-content')

# Example 4: Perform scraping with additional parameters for a customized experience
tool = SeleniumScrapingTool(website_url='https://example.com', css_element='.main-content', cookie={'name': 'user', 'value': 'John Doe'}, wait_time=10)
```

## Arguments
The following parameters can be used to customize the SeleniumScrapingTool's scraping process:

- `website_url`: **Mandatory**. Specifies the URL of the website from which content is to be scraped.
- `css_element`: **Mandatory**. The CSS selector for a specific element to target on the website. This enables focused scraping of a particular part of a webpage.
- `cookie`: **Optional**. A dictionary that contains cookie information. Useful for simulating a logged-in session, thereby providing access to content that might be restricted to non-logged-in users.
- `wait_time`: **Optional**. Specifies the delay (in seconds) before the content is scraped. This delay allows for the website and any dynamic content to fully load, ensuring a successful scrape.

!!! attention
    Since the SeleniumScrapingTool is under active development, the parameters and functionality may evolve over time. Users are encouraged to keep the tool updated and report any issues or suggestions for enhancements.

---


# SerperDevTool.md

# SerperDevTool Documentation

!!! note "Experimental"
    We are still working on improving tools, so there might be unexpected behavior or changes in the future.

## Description
This tool is designed to perform a semantic search for a specified query from a text's content across the internet. It utilizes the [serper.dev](https://serper.dev) API to fetch and display the most relevant search results based on the query provided by the user.

## Installation
To incorporate this tool into your project, follow the installation instructions below:
```shell
pip install 'crewai[tools]'
```

## Example
The following example demonstrates how to initialize the tool and execute a search with a given query:

```python
from crewai_tools import SerperDevTool

# Initialize the tool for internet searching capabilities
tool = SerperDevTool()
```

## Steps to Get Started
To effectively use the `SerperDevTool`, follow these steps:

1. **Package Installation**: Confirm that the `crewai[tools]` package is installed in your Python environment.
2. **API Key Acquisition**: Acquire a `serper.dev` API key by registering for a free account at `serper.dev`.
3. **Environment Configuration**: Store your obtained API key in an environment variable named `SERPER_API_KEY` to facilitate its use by the tool.

## Parameters

The `SerperDevTool` comes with several parameters that will be passed to the API :

- **search_url**: The URL endpoint for the search API. (Default is `https://google.serper.dev/search`)

- **country**: Optional. Specify the country for the search results.
- **location**: Optional. Specify the location for the search results.
- **locale**: Optional. Specify the locale for the search results.
- **n_results**: Number of search results to return. Default is `10`.

The values for `country`, `location`, `locale` and `search_url` can be found on the [Serper Playground](https://serper.dev/playground).

## Example with Parameters
Here is an example demonstrating how to use the tool with additional parameters:

```python
from crewai_tools import SerperDevTool

tool = SerperDevTool(
    search_url="https://google.serper.dev/scholar",
    n_results=2,
)

print(tool.run(search_query="ChatGPT"))

# Using Tool: Search the internet

# Search results: Title: Role of chat gpt in public health
# Link: https://link.springer.com/article/10.1007/s10439-023-03172-7
# Snippet: … ChatGPT in public health. In this overview, we will examine the potential uses of ChatGPT in
# ---
# Title: Potential use of chat gpt in global warming
# Link: https://link.springer.com/article/10.1007/s10439-023-03171-8
# Snippet: … as ChatGPT, have the potential to play a critical role in advancing our understanding of climate
# ---

```

```python
from crewai_tools import SerperDevTool

tool = SerperDevTool(
    country="fr",
    locale="fr",
    location="Paris, Paris, Ile-de-France, France",
    n_results=2,
)

print(tool.run(search_query="Jeux Olympiques"))

# Using Tool: Search the internet

# Search results: Title: Jeux Olympiques de Paris 2024 - Actualités, calendriers, résultats
# Link: https://olympics.com/fr/paris-2024
# Snippet: Quels sont les sports présents aux Jeux Olympiques de Paris 2024 ? · Athlétisme · Aviron · Badminton · Basketball · Basketball 3x3 · Boxe · Breaking · Canoë ...
# ---
# Title: Billetterie Officielle de Paris 2024 - Jeux Olympiques et Paralympiques
# Link: https://tickets.paris2024.org/
# Snippet: Achetez vos billets exclusivement sur le site officiel de la billetterie de Paris 2024 pour participer au plus grand événement sportif au monde.
# ---

```

## Conclusion
By integrating the `SerperDevTool` into Python projects, users gain the ability to conduct real-time, relevant searches across the internet directly from their applications. The updated parameters allow for more customized and localized search results. By adhering to the setup and usage guidelines provided, incorporating this tool into projects is streamlined and straightforward.


---


# TXTSearchTool.md

# TXTSearchTool

!!! note "Experimental"
    We are still working on improving tools, so there might be unexpected behavior or changes in the future.

## Description
This tool is used to perform a RAG (Retrieval-Augmented Generation) search within the content of a text file. It allows for semantic searching of a query within a specified text file's content, making it an invaluable resource for quickly extracting information or finding specific sections of text based on the query provided.

## Installation
To use the TXTSearchTool, you first need to install the crewai_tools package. This can be done using pip, a package manager for Python. Open your terminal or command prompt and enter the following command:

```shell
pip install 'crewai[tools]'
```

This command will download and install the TXTSearchTool along with any necessary dependencies.

## Example
The following example demonstrates how to use the TXTSearchTool to search within a text file. This example shows both the initialization of the tool with a specific text file and the subsequent search within that file's content.

```python
from crewai_tools import TXTSearchTool

# Initialize the tool to search within any text file's content the agent learns about during its execution
tool = TXTSearchTool()

# OR

# Initialize the tool with a specific text file, so the agent can search within the given text file's content
tool = TXTSearchTool(txt='path/to/text/file.txt')
```

## Arguments
- `txt` (str): **Optional**. The path to the text file you want to search. This argument is only required if the tool was not initialized with a specific text file; otherwise, the search will be conducted within the initially provided text file.

## Custom model and embeddings

By default, the tool uses OpenAI for both embeddings and summarization. To customize the model, you can use a config dictionary as follows:

```python
tool = TXTSearchTool(
    config=dict(
        llm=dict(
            provider="ollama", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="llama2",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="google", # or openai, ollama, ...
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)
```


---


# WebsiteSearchTool.md

# WebsiteSearchTool

!!! note "Experimental Status"
    The WebsiteSearchTool is currently in an experimental phase. We are actively working on incorporating this tool into our suite of offerings and will update the documentation accordingly.

## Description
The WebsiteSearchTool is designed as a concept for conducting semantic searches within the content of websites. It aims to leverage advanced machine learning models like Retrieval-Augmented Generation (RAG) to navigate and extract information from specified URLs efficiently. This tool intends to offer flexibility, allowing users to perform searches across any website or focus on specific websites of interest. Please note, the current implementation details of the WebsiteSearchTool are under development, and its functionalities as described may not yet be accessible.

## Installation
To prepare your environment for when the WebsiteSearchTool becomes available, you can install the foundational package with:

```shell
pip install 'crewai[tools]'
```

This command installs the necessary dependencies to ensure that once the tool is fully integrated, users can start using it immediately.

## Example Usage
Below are examples of how the WebsiteSearchTool could be utilized in different scenarios. Please note, these examples are illustrative and represent planned functionality:

```python
from crewai_tools import WebsiteSearchTool

# Example of initiating tool that agents can use to search across any discovered websites
tool = WebsiteSearchTool()

# Example of limiting the search to the content of a specific website, so now agents can only search within that website
tool = WebsiteSearchTool(website='https://example.com')
```

## Arguments
- `website`: An optional argument intended to specify the website URL for focused searches. This argument is designed to enhance the tool's flexibility by allowing targeted searches when necessary.

## Customization Options
By default, the tool uses OpenAI for both embeddings and summarization. To customize the model, you can use a config dictionary as follows:


```python
tool = WebsiteSearchTool(
    config=dict(
        llm=dict(
            provider="ollama", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="llama2",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="google", # or openai, ollama, ...
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)
```

---


# XMLSearchTool.md

# XMLSearchTool

!!! note "Experimental"
    We are still working on improving tools, so there might be unexpected behavior or changes in the future.

## Description
The XMLSearchTool is a cutting-edge RAG tool engineered for conducting semantic searches within XML files. Ideal for users needing to parse and extract information from XML content efficiently, this tool supports inputting a search query and an optional XML file path. By specifying an XML path, users can target their search more precisely to the content of that file, thereby obtaining more relevant search outcomes.

## Installation
To start using the XMLSearchTool, you must first install the crewai_tools package. This can be easily done with the following command:

```shell
pip install 'crewai[tools]'
```

## Example
Here are two examples demonstrating how to use the XMLSearchTool. The first example shows searching within a specific XML file, while the second example illustrates initiating a search without predefining an XML path, providing flexibility in search scope.

```python
from crewai_tools import XMLSearchTool

# Allow agents to search within any XML file's content as it learns about their paths during execution
tool = XMLSearchTool()

# OR

# Initialize the tool with a specific XML file path for exclusive search within that document
tool = XMLSearchTool(xml='path/to/your/xmlfile.xml')
```

## Arguments
- `xml`: This is the path to the XML file you wish to search. It is an optional parameter during the tool's initialization but must be provided either at initialization or as part of the `run` method's arguments to execute a search.

## Custom model and embeddings

By default, the tool uses OpenAI for both embeddings and summarization. To customize the model, you can use a config dictionary as follows:

```python
tool = XMLSearchTool(
    config=dict(
        llm=dict(
            provider="ollama", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="llama2",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="google", # or openai, ollama, ...
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)
```


---


# YoutubeChannelSearchTool.md

# YoutubeChannelSearchTool

!!! note "Experimental"
    We are still working on improving tools, so there might be unexpected behavior or changes in the future.

## Description
This tool is designed to perform semantic searches within a specific Youtube channel's content. Leveraging the RAG (Retrieval-Augmented Generation) methodology, it provides relevant search results, making it invaluable for extracting information or finding specific content without the need to manually sift through videos. It streamlines the search process within Youtube channels, catering to researchers, content creators, and viewers seeking specific information or topics.

## Installation
To utilize the YoutubeChannelSearchTool, the `crewai_tools` package must be installed. Execute the following command in your shell to install:

```shell
pip install 'crewai[tools]'
```

## Example
To begin using the YoutubeChannelSearchTool, follow the example below. This demonstrates initializing the tool with a specific Youtube channel handle and conducting a search within that channel's content.

```python
from crewai_tools import YoutubeChannelSearchTool

# Initialize the tool to search within any Youtube channel's content the agent learns about during its execution
tool = YoutubeChannelSearchTool()

# OR

# Initialize the tool with a specific Youtube channel handle to target your search
tool = YoutubeChannelSearchTool(youtube_channel_handle='@exampleChannel')
```

## Arguments
- `youtube_channel_handle` : A mandatory string representing the Youtube channel handle. This parameter is crucial for initializing the tool to specify the channel you want to search within. The tool is designed to only search within the content of the provided channel handle.

## Custom model and embeddings

By default, the tool uses OpenAI for both embeddings and summarization. To customize the model, you can use a config dictionary as follows:

```python
tool = YoutubeChannelSearchTool(
    config=dict(
        llm=dict(
            provider="ollama", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="llama2",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="google", # or openai, ollama, ...
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)
```


---


# YoutubeVideoSearchTool.md

# YoutubeVideoSearchTool

!!! note "Experimental"
    We are still working on improving tools, so there might be unexpected behavior or changes in the future.

## Description

This tool is part of the `crewai_tools` package and is designed to perform semantic searches within Youtube video content, utilizing Retrieval-Augmented Generation (RAG) techniques. It is one of several "Search" tools in the package that leverage RAG for different sources. The YoutubeVideoSearchTool allows for flexibility in searches; users can search across any Youtube video content without specifying a video URL, or they can target their search to a specific Youtube video by providing its URL.

## Installation

To utilize the YoutubeVideoSearchTool, you must first install the `crewai_tools` package. This package contains the YoutubeVideoSearchTool among other utilities designed to enhance your data analysis and processing tasks. Install the package by executing the following command in your terminal:

```
pip install 'crewai[tools]'
```

## Example

To integrate the YoutubeVideoSearchTool into your Python projects, follow the example below. This demonstrates how to use the tool both for general Youtube content searches and for targeted searches within a specific video's content.

```python
from crewai_tools import YoutubeVideoSearchTool

# General search across Youtube content without specifying a video URL, so the agent can search within any Youtube video content it learns about irs url during its operation
tool = YoutubeVideoSearchTool()

# Targeted search within a specific Youtube video's content
tool = YoutubeVideoSearchTool(youtube_video_url='https://youtube.com/watch?v=example')
```

## Arguments

The YoutubeVideoSearchTool accepts the following initialization arguments:

- `youtube_video_url`: An optional argument at initialization but required if targeting a specific Youtube video. It specifies the Youtube video URL path you want to search within.

## Custom model and embeddings

By default, the tool uses OpenAI for both embeddings and summarization. To customize the model, you can use a config dictionary as follows:

```python
tool = YoutubeVideoSearchTool(
    config=dict(
        llm=dict(
            provider="ollama", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="llama2",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="google", # or openai, ollama, ...
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)
```


---
