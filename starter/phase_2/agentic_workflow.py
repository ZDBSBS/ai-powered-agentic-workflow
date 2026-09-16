# agentic_workflow.py

# TODO: 1 - Import the following agents: ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent from the workflow_agents.base_agents module
from workflow_agents.base_agents import (
    ActionPlanningAgent,
    EvaluationAgent,
    KnowledgeAugmentedPromptAgent,
    RoutingAgent,
)

import os
from dotenv import load_dotenv

# TODO: 2 - Load the OpenAI key into a variable called openai_api_key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# load the product spec
# TODO: 3 - Load the product spec document Product-Spec-Email-Router.txt into a variable called product_spec
with open(
    "Product-Spec-Email-Router.txt",
    "r",
    encoding="utf-8"
) as product_spec_file:
    product_spec = product_spec_file.read()

# Instantiate all the agents

# Action Planning Agent
knowledge_action_planning = (
    "A complete product development plan consists of exactly three major "
    "workflow steps, executed in this order:\n"
    "1. Define user stories from the provided product specification. Each "
    "user story must identify a user persona, a desired action or feature, "
    "and the resulting benefit.\n"
    "2. Define product features from the provided product specification by "
    "organizing related product capabilities into cohesive feature groups.\n"
    "3. Define detailed engineering tasks from the provided product "
    "specification for implementing the user stories and product features.\n"
    "When creating an action plan, return exactly these three independent "
    "and actionable steps. Do not divide them into smaller substeps and do "
    "not duplicate any step."
)

# TODO: 4 - Instantiate an action_planning_agent using the 'knowledge_action_planning'
action_planning_agent = ActionPlanningAgent(
    openai_api_key,
    knowledge_action_planning
)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = (
    "You are a Product Manager, you are responsible for defining the "
    "user stories for a product."
)

knowledge_product_manager = (
    "Define user stories only for the product described in the product "
    "specification below.\n"
    "Each user story must follow this exact structure:\n"
    "As a [type of user], I want [an action or feature] so that "
    "[benefit/value].\n"
    "Identify the relevant user personas, actions, and benefits directly "
    "from the product specification. Generate several distinct user stories. "
    "Do not define product features or engineering tasks. Do not introduce "
    "capabilities that are not supported by the product specification.\n\n"
    "Product specification:\n"
    # TODO: 5 - Complete this knowledge string by appending the product_spec loaded in TODO 3
    + product_spec
)

# TODO: 6 - Instantiate a product_manager_knowledge_agent using 'persona_product_manager' and the completed 'knowledge_product_manager'
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    persona_product_manager,
    knowledge_product_manager
)

# Product Manager - Evaluation Agent
# TODO: 7 - Define the persona and evaluation criteria for a Product Manager evaluation agent and instantiate it as product_manager_evaluation_agent. This agent will evaluate the product_manager_knowledge_agent.
# The evaluation_criteria should specify the expected structure for user stories (e.g., "As a [type of user], I want [an action or feature] so that [benefit/value].").
persona_product_manager_eval = (
    "You are an evaluation agent that checks the answers of other "
    "worker agents"
)

evaluation_criteria_product_manager = (
    "The answer must contain one or more user stories. "
    "Every user story must follow this exact structure:\n"
    "As a [type of user], I want [an action or feature] so that "
    "[benefit/value].\n"
    "Every story must clearly contain a user persona, a desired action or "
    "feature, and a user benefit or value. Reject the answer if any story "
    "does not follow this structure or contains functionality unrelated to "
    "the product described in the provided product specification."
)

product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_product_manager_eval,
    evaluation_criteria_product_manager,
    product_manager_knowledge_agent,
    10
)

# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = (
    "You are a Program Manager, you are responsible for defining the "
    "features for a product."
)

knowledge_program_manager = (
    "Define product features only for the product described in the product "
    "specification below.\n"
    "Derive every feature directly from related capabilities and user needs "
    "contained in the specification. Do not introduce unrelated or generic "
    "features.\n"
    "Every feature must use the following exact structure:\n"
    "Feature Name: A clear and concise capability name\n"
    "Description: The purpose of the feature\n"
    "Key Functionality: The specific capabilities provided by the feature\n"
    "User Benefit: The value created for the user\n"
    "Do not define user profiles, social sharing, shopping carts, or other "
    "capabilities unless they are explicitly supported by the specification.\n\n"
    "Product specification:\n"
    + product_spec
)

# Instantiate a program_manager_knowledge_agent using 'persona_program_manager' and 'knowledge_program_manager'
# (This is a necessary step before TODO 8. Students should add the instantiation code here.)
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    persona_program_manager,
    knowledge_program_manager
)

# Program Manager - Evaluation Agent
persona_program_manager_eval = (
    "You are an evaluation agent that checks the answers of other "
    "worker agents."
)

# TODO: 8 - Instantiate a program_manager_evaluation_agent using 'persona_program_manager_eval' and the evaluation criteria below.
#                      "The answer should be product features that follow the following structure: " \
#                      "Feature Name: A clear, concise title that identifies the capability\n" \
#                      "Description: A brief explanation of what the feature does and its purpose\n" \
#                      "Key Functionality: The specific capabilities or actions the feature provides\n" \
#                      "User Benefit: How this feature creates value for the user"
# For the 'agent_to_evaluate' parameter, refer to the provided solution code's pattern.
evaluation_criteria_program_manager = (
    "The answer must contain one or more product features. "
    "Every feature must use all four exact field names below, in this "
    "exact order:\n"
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its "
    "purpose\n"
    "Key Functionality: The specific capabilities or actions the feature "
    "provides\n"
    "User Benefit: How this feature creates value for the user\n"
    "Reject the answer if any feature is missing one of these fields, "
    "uses a different field name, or contains functionality unrelated to "
    "the product described in the provided product specification."
)

program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_program_manager_eval,
    evaluation_criteria_program_manager,
    program_manager_knowledge_agent,
    10
)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = (
    "You are a Development Engineer, you are responsible for defining the "
    "development tasks for a product."
)

knowledge_dev_engineer = (
    "Development tasks are concrete, actionable units of engineering work "
    "derived from the user stories, product features, and requirements in "
    "the provided product specification. Generate actual implementation "
    "tasks, not instructions, examples, templates, or explanations of how "
    "tasks should be written. Every task must begin with the literal field "
    "name 'Task ID:' followed by a unique identifier. Do not use headings "
    "such as 'Task 1' as a replacement for 'Task ID:'. Every task must "
    "contain the following exact field names in this exact order:\n"
    "Task ID: A unique identifier\n"
    "Task Title: A concise description of the implementation work\n"
    "Related User Story: A complete parent user story using the format "
    "'As a [type of user], I want [an action or feature] so that "
    "[benefit/value].'\n"
    "Description: The detailed technical work required\n"
    "Acceptance Criteria: Specific and verifiable completion requirements\n"
    "Estimated Effort: A time or complexity estimate\n"
    "Dependencies: Tasks or conditions that must be completed first\n"
    "All seven fields must contain actual values. The Related User Story "
    "field must contain a complete and relevant user story. Do not use "
    "'N/A', 'None', a feature name, or a generic placeholder for this "
    "field. Use 'None' for Dependencies only when a task has no dependency. "
    "Do not return instructions or descriptions of the required format. "
    "Use only information from the provided product specification. Do not "
    "invent unrelated product capabilities, users, requirements, or tasks:\n"
    + product_spec
)

# Instantiate a development_engineer_knowledge_agent using 'persona_dev_engineer' and 'knowledge_dev_engineer'
# (This is a necessary step before TODO 9. Students should add the instantiation code here.)
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    persona_dev_engineer,
    knowledge_dev_engineer
)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = (
    "You are an evaluation agent that checks the answers of other "
    "worker agents."
)

# TODO: 9 - Instantiate a development_engineer_evaluation_agent using 'persona_dev_engineer_eval' and the evaluation criteria below.
#                      "The answer should be tasks following this exact structure: " \
#                      "Task ID: A unique identifier for tracking purposes\n" \
#                      "Task Title: Brief description of the specific development work\n" \
#                      "Related User Story: Reference to the parent user story\n" \
#                      "Description: Detailed explanation of the technical work required\n" \
#                      "Acceptance Criteria: Specific requirements that must be met for completion\n" \
#                      "Estimated Effort: Time or complexity estimation\n" \
#                      "Dependencies: Any tasks that must be completed first"
# For the 'agent_to_evaluate' parameter, refer to the provided solution code's pattern.
evaluation_criteria_dev_engineer = (
    "The answer must contain a comprehensive list of multiple engineering "
    "tasks for implementing the product described in the provided product "
    "specification. Reject an answer that contains only one task when the "
    "specification clearly requires multiple implementation areas.\n"
    "Every task must use all seven exact field names below, in this exact "
    "order:\n"
    "Task ID: A unique identifier for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n"
    "Related User Story: Reference to the parent user story\n"
    "Description: Detailed explanation of the technical work required\n"
    "Acceptance Criteria: Specific requirements that must be met for "
    "completion\n"
    "Estimated Effort: Time or complexity estimation\n"
    "Dependencies: Any tasks that must be completed first\n"
    "Every field must contain a concrete value. Task IDs must be unique. "
    "The tasks must be actionable and collectively cover the major product "
    "capabilities represented in the supplied input and specification.\n"
    "Reject the answer if valid tasks from the supplied input were discarded "
    "without reason, if the response is merely an acknowledgement, template, "
    "example, instruction, or explanation, or if any task is unrelated to "
    "the product described in the provided product specification."
)

development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_dev_engineer_eval,
    evaluation_criteria_dev_engineer,
    development_engineer_knowledge_agent,
    10
)


# Routing Agent
# TODO: 10 - Instantiate a routing_agent. You will need to define a list of agent dictionaries (routes) for Product Manager, Program Manager, and Development Engineer. Each dictionary should contain 'name', 'description', and 'func' (linking to a support function). Assign this list to the routing_agent's 'agents' attribute.
routing_agent = RoutingAgent(openai_api_key, {})

agents = [
    {
        "name": "Product Manager",
        "description": (
            "Responsible only for identifying product user personas and "
            "defining user stories from a product specification. User stories "
            "contain a user type, desired action or feature, and resulting "
            "benefit. Does not define features, group stories, or create "
            "engineering tasks."
        ),
        "func": lambda query: product_manager_support_function(query)
    },
    {
        "name": "Program Manager",
        "description": (
            "Responsible only for defining product features and grouping "
            "related user stories or product capabilities into cohesive "
            "features. Does not define user personas, write user stories, "
            "or create engineering tasks."
        ),
        "func": lambda query: program_manager_support_function(query)
    },
    {
        "name": "Development Engineer",
        "description": (
            "Responsible only for defining detailed technical engineering "
            "tasks required to implement user stories and product features. "
            "Defines acceptance criteria, effort, and dependencies. Does not "
            "define user personas, user stories, or product features."
        ),
        "func": lambda query: development_engineer_support_function(query)
    }
]

routing_agent.agents = agents

# Job function persona support functions
# TODO: 11 - Define the support functions for the routes of the routing agent (e.g., product_manager_support_function, program_manager_support_function, development_engineer_support_function).
# Each support function should:
#   1. Take the input query (e.g., a step from the action plan).
#   2. Get a response from the respective Knowledge Augmented Prompt Agent.
#   3. Have the response evaluated by the corresponding Evaluation Agent.
#   4. Return the final validated response.
def product_manager_support_function(query):
    response_from_knowledge_agent = (
        product_manager_knowledge_agent.respond(query)
    )
    evaluation_result = product_manager_evaluation_agent.evaluate(
        response_from_knowledge_agent
    )
    return evaluation_result["final_response"]


def program_manager_support_function(query):
    response_from_knowledge_agent = (
        program_manager_knowledge_agent.respond(query)
    )
    evaluation_result = program_manager_evaluation_agent.evaluate(
        response_from_knowledge_agent
    )
    return evaluation_result["final_response"]


def development_engineer_support_function(query):
    response_from_knowledge_agent = (
        development_engineer_knowledge_agent.respond(query)
    )
    evaluation_result = development_engineer_evaluation_agent.evaluate(
        response_from_knowledge_agent
    )
    return evaluation_result["final_response"]


# Run the workflow

print("\n*** Workflow execution started ***\n")

# Workflow Prompt
# ****
workflow_prompt = (
    "Create a complete development plan for the product described in the "
    "provided product specification. Produce exactly three major workflow "
    "steps. First, define user stories based on the product specification. "
    "Second, define product features based on related product capabilities "
    "and user needs from the specification. Third, define detailed "
    "engineering tasks for implementing the user stories and product "
    "features. Do not divide these three major steps into smaller or "
    "duplicate steps."
)
# ****

print(
    f"Task to complete in this workflow, workflow prompt = {workflow_prompt}"
)

print("\nDefining workflow steps from the workflow prompt")

# TODO: 12 - Implement the workflow.
#   1. Use the 'action_planning_agent' to extract steps from the 'workflow_prompt'.
#   2. Initialize an empty list to store 'completed_steps'.
#   3. Loop through the extracted workflow steps:
#      a. For each step, use the 'routing_agent' to route the step to the appropriate support function.
#      b. Append the result to 'completed_steps'.
#      c. Print information about the step being executed and its result.
#   4. After the loop, print the final output of the workflow (the last completed step).
workflow_steps = action_planning_agent.extract_steps_from_prompt(
    workflow_prompt
)
completed_steps = []

for step in workflow_steps:
    print(f"\nExecuting workflow step:\n{step}")
    step_result = routing_agent.route(step)
    completed_steps.append(step_result)
    print(f"\nWorkflow step result:\n{step_result}")

if completed_steps:
    print("\n*** Final workflow output ***\n")

    for step_number, completed_step in enumerate(
        completed_steps,
        start=1
    ):
        print(f"\n--- Completed Step {step_number} ---\n")
        print(completed_step)
else:
    print("\nNo workflow steps were completed.")