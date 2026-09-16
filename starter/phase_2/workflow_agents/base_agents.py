# TODO: 1 - import the OpenAI class from the openai library
from openai import OpenAI
import ast
import csv
import re
import uuid
from datetime import datetime

import numpy as np
import pandas as pd


OPENAI_BASE_URL = "https://openai.vocareum.com/v1"


# DirectPromptAgent class definition
class DirectPromptAgent:

    def __init__(self, openai_api_key):
        # Initialize the agent
        # TODO: 2 - Define an attribute named openai_api_key to store the OpenAI API key provided to this class.
        self.openai_api_key = openai_api_key

    def respond(self, prompt):
        # Generate a response using the OpenAI API
        client = OpenAI(
            base_url=OPENAI_BASE_URL,
            api_key=self.openai_api_key
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # TODO: 3 - Specify the model to use (gpt-3.5-turbo)
            messages=[
                # TODO: 4 - Provide the user's prompt here. Do not add a system prompt.
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        # TODO: 5 - Return only the textual content of the response (not the full JSON response).
        return response.choices[0].message.content


# AugmentedPromptAgent class definition
class AugmentedPromptAgent:

    def __init__(self, openai_api_key, persona):
        """Initialize the agent with given attributes."""
        # TODO: 1 - Create an attribute for the agent's persona
        self.persona = persona
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """Generate a response using OpenAI API."""
        client = OpenAI(
            base_url=OPENAI_BASE_URL,
            api_key=self.openai_api_key
        )

        # TODO: 2 - Declare a variable 'response' that calls OpenAI's API for a chat completion.
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                # TODO: 3 - Add a system prompt instructing the agent to assume the defined persona and explicitly forget previous context.
                {
                    "role": "system",
                    "content": (
                        f"{self.persona}\n"
                        "Forget all previous conversational context. "
                        "Follow the defined persona and its response-format "
                        "instructions exactly."
                    )
                },
                {
                    "role": "user",
                    "content": input_text
                }
            ],
            temperature=0
        )

        # TODO: 4 - Return only the textual content of the response, not the full JSON payload.
        return response.choices[0].message.content


# KnowledgeAugmentedPromptAgent class definition
class KnowledgeAugmentedPromptAgent:

    def __init__(self, openai_api_key, persona, knowledge):
        """Initialize the agent with provided attributes."""
        self.persona = persona

        # TODO: 1 - Create an attribute to store the agent's knowledge.
        self.knowledge = knowledge

        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """Generate a response using the OpenAI API."""
        client = OpenAI(
            base_url=OPENAI_BASE_URL,
            api_key=self.openai_api_key
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                # TODO: 2 - Construct a system message including:
                #           - The persona with the following instruction:
                #             "You are _persona_ knowledge-based assistant. Forget all previous context."
                #           - The provided knowledge with this instruction:
                #             "Use only the following knowledge to answer, do not use your own knowledge: _knowledge_"
                #           - Final instruction:
                #             "Answer the prompt based on this knowledge, not your own."
                {
                    "role": "system",
                    "content": (
                        f"{self.persona}\n"
                        "You are a knowledge-based assistant. "
                        "Forget all previous context.\n"
                        "Use only the following knowledge to answer. "
                        "Do not use your own knowledge:\n"
                        f"{self.knowledge}\n"
                        "Answer the prompt based on this knowledge, not your "
                        "own. Follow the persona and its response-format "
                        "instructions exactly. Return the requested completed "
                        "content directly. Do not merely acknowledge, describe, "
                        "summarize, or repeat the request."
                    )
                },

                # TODO: 3 - Add the user's input prompt here as a user message.
                {
                    "role": "user",
                    "content": input_text
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content


# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge
    from a large corpus and leverages embeddings to respond to prompts based
    solely on retrieved information.
    """

    def __init__(
        self,
        openai_api_key,
        persona,
        chunk_size=2000,
        chunk_overlap=100
    ):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and
        configuration settings.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding.
        chunk_overlap (int): Overlap between consecutive chunks.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
            f"{uuid.uuid4().hex[:8]}.csv"
        )

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding
        API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        client = OpenAI(
            base_url=OPENAI_BASE_URL,
            api_key=self.openai_api_key
        )

        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )

        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1 = np.array(vector_one)
        vec2 = np.array(vector_two)

        denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)

        if denominator == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / denominator)

    def chunk_text(self, text):
        """
        Splits text into manageable chunks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return []

        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start,
                "start_char": start,
                "end_char": end
            })

            if end >= len(text):
                break

            next_start = end - self.chunk_overlap

            if next_start <= start:
                raise RuntimeError(
                    "Chunking did not advance. Check chunk_size and "
                    "chunk_overlap."
                )

            start = next_start
            chunk_id += 1

        with open(
            f"chunks-{self.unique_filename}",
            "w",
            newline="",
            encoding="utf-8"
        ) as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=["text", "chunk_size"]
            )
            writer.writeheader()

            for chunk in chunks:
                writer.writerow({
                    "text": chunk["text"],
                    "chunk_size": chunk["chunk_size"]
                })

        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and embeddings.
        """
        df = pd.read_csv(
            f"chunks-{self.unique_filename}",
            encoding="utf-8"
        )

        df["embeddings"] = df["text"].apply(self.get_embedding)

        df.to_csv(
            f"embeddings-{self.unique_filename}",
            encoding="utf-8",
            index=False
        )

        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded
        knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)

        df = pd.read_csv(
            f"embeddings-{self.unique_filename}",
            encoding="utf-8"
        )

        df["embeddings"] = df["embeddings"].apply(
            lambda value: np.array(ast.literal_eval(value))
        )

        df["similarity"] = df["embeddings"].apply(
            lambda embedding: self.calculate_similarity(
                prompt_embedding,
                embedding
            )
        )

        best_chunk = df.loc[df["similarity"].idxmax(), "text"]

        client = OpenAI(
            base_url=OPENAI_BASE_URL,
            api_key=self.openai_api_key
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are {self.persona}, a knowledge-based "
                        "assistant. Forget previous context."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Answer based only on this information:\n"
                        f"{best_chunk}\n\n"
                        f"Prompt:\n{prompt}"
                    )
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content


class EvaluationAgent:

    def __init__(
        self,
        openai_api_key,
        persona,
        evaluation_criteria,
        worker_agent,
        max_interactions
    ):
        # Initialize the EvaluationAgent with given attributes.
        # TODO: 1 - Declare class attributes here
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions

    def _validate_user_stories(self, answer):
        """
        Deterministically validates user-story output.
        """
        non_empty_lines = [
            line.strip()
            for line in answer.splitlines()
            if line.strip()
        ]

        story_lines = [
            line
            for line in non_empty_lines
            if re.match(r"^as an? ", line.lower())
        ]

        if not story_lines:
            return (
                False,
                "The answer does not contain any user story beginning with "
                "'As a'."
            )

        invalid_stories = []

        for story in story_lines:
            normalized_story = " ".join(story.split()).lower()

            if (
                " i want " not in normalized_story
                or " so that " not in normalized_story
            ):
                invalid_stories.append(story)

        if invalid_stories:
            return (
                False,
                "One or more user stories do not contain the complete "
                "'As a ..., I want ..., so that ...' structure."
            )

        unrelated_lines = [
            line
            for line in non_empty_lines
            if re.match(r"^as an? ", line.lower()) is None
        ]

        if unrelated_lines:
            return (
                False,
                "The answer contains content outside the required user-story "
                "format."
            )

        return True, "All user stories use the required structure."

    def _validate_features(self, answer):
        """
        Deterministically validates product-feature output.
        """
        required_fields = [
            "Feature Name:",
            "Description:",
            "Key Functionality:",
            "User Benefit:"
        ]

        feature_count = answer.count("Feature Name:")

        if feature_count < 1:
            return (
                False,
                "The answer does not contain any 'Feature Name:' field."
            )

        field_counts = {
            field: answer.count(field)
            for field in required_fields
        }

        for field, count in field_counts.items():
            if count != feature_count:
                return (
                    False,
                    f"The number of '{field}' fields does not match the "
                    "number of product features."
                )

        feature_sections = answer.split("Feature Name:")[1:]

        for section_number, section in enumerate(
            feature_sections,
            start=1
        ):
            positions = []

            for field in required_fields[1:]:
                position = section.find(field)

                if position == -1:
                    return (
                        False,
                        f"Feature {section_number} is missing '{field}'."
                    )

                positions.append(position)

            if positions != sorted(positions):
                return (
                    False,
                    f"Feature {section_number} does not use the required "
                    "field order."
                )

            feature_name = section.split(
                "Description:",
                maxsplit=1
            )[0].strip()

            description = section.split(
                "Description:",
                maxsplit=1
            )[1].split(
                "Key Functionality:",
                maxsplit=1
            )[0].strip()

            functionality = section.split(
                "Key Functionality:",
                maxsplit=1
            )[1].split(
                "User Benefit:",
                maxsplit=1
            )[0].strip()

            benefit = section.split(
                "User Benefit:",
                maxsplit=1
            )[1].strip()

            if not all([
                feature_name,
                description,
                functionality,
                benefit
            ]):
                return (
                    False,
                    f"Feature {section_number} contains an empty required "
                    "field."
                )

        return (
            True,
            "All product features contain the required fields in order."
        )

    def _validate_engineering_tasks(self, answer):
        """
        Deterministically validates engineering-task output.
        """
        required_fields = [
            "Task ID:",
            "Task Title:",
            "Related User Story:",
            "Description:",
            "Acceptance Criteria:",
            "Estimated Effort:",
            "Dependencies:"
        ]

        task_start_pattern = re.compile(
            r"(?mi)^\s*(?:\*\*)?Task ID:(?:\*\*)?\s*(?P<value>.+?)\s*$"
        )

        task_matches = list(task_start_pattern.finditer(answer))
        task_count = len(task_matches)

        if task_count < 2:
            return (
                False,
                "The answer must contain multiple engineering tasks. Fewer "
                "than two task sections beginning with 'Task ID:' were found."
            )

        task_sections = []

        for index, task_match in enumerate(task_matches):
            section_start = task_match.start()

            if index + 1 < task_count:
                section_end = task_matches[index + 1].start()
            else:
                section_end = len(answer)

            task_sections.append(
                answer[section_start:section_end]
            )

        task_ids = []

        invalid_values = {
            "n/a",
            "not applicable",
            "placeholder",
            "to be determined",
            "tbd"
        }

        for section_number, section in enumerate(
            task_sections,
            start=1
        ):
            field_values = {}
            field_positions = []

            for field in required_fields:
                field_name = field[:-1]

                field_pattern = re.compile(
                    rf"(?mi)^\s*(?:\*\*)?"
                    rf"{re.escape(field_name)}:"
                    rf"(?:\*\*)?\s*(?P<value>.+?)\s*$"
                )

                field_matches = list(
                    field_pattern.finditer(section)
                )

                if not field_matches:
                    return (
                        False,
                        f"Task {section_number} is missing '{field}'."
                    )

                if len(field_matches) > 1:
                    return (
                        False,
                        f"Task {section_number} contains more than one "
                        f"'{field}' field."
                    )

                field_match = field_matches[0]

                field_positions.append(
                    field_match.start()
                )

                field_values[field] = (
                    field_match.group("value").strip()
                )

            if field_positions != sorted(field_positions):
                return (
                    False,
                    f"Task {section_number} does not use the required field "
                    "order."
                )

            empty_fields = [
                field
                for field, value in field_values.items()
                if not value
            ]

            if empty_fields:
                return (
                    False,
                    f"Task {section_number} contains an empty required field: "
                    f"{empty_fields[0]}"
                )

            task_id = field_values["Task ID:"]
            task_title = field_values["Task Title:"]
            related_story = field_values["Related User Story:"]
            description = field_values["Description:"]
            acceptance_criteria = field_values["Acceptance Criteria:"]
            estimated_effort = field_values["Estimated Effort:"]
            dependencies = field_values["Dependencies:"]

            if task_id in task_ids:
                return (
                    False,
                    f"Task ID '{task_id}' is duplicated."
                )

            task_ids.append(task_id)

            required_task_values = [
                task_id,
                task_title,
                related_story,
                description,
                acceptance_criteria,
                estimated_effort
            ]

            for value in required_task_values:
                if value.strip().lower() in invalid_values:
                    return (
                        False,
                        f"Task {section_number} contains a placeholder value."
                    )

            if dependencies.strip().lower() in invalid_values:
                return (
                    False,
                    f"Task {section_number} contains an invalid Dependencies "
                    "value. Use 'None' when no dependency exists."
                )

        return (
            True,
            "All engineering tasks contain the seven required fields, "
            "unique IDs, concrete values, and related user story references."
        )

    def _run_deterministic_validation(self, answer):
        """
        Selects a deterministic validation based on the evaluation criteria.
        """
        criteria_lower = self.evaluation_criteria.lower()

        task_markers = [
            "task id:",
            "task title:",
            "related user story:",
            "acceptance criteria:",
            "estimated effort:",
            "dependencies:"
        ]

        feature_markers = [
            "feature name:",
            "key functionality:",
            "user benefit:"
        ]

        user_story_markers = [
            "as a [type of user]",
            "i want [an action or feature]",
            "so that [benefit/value]"
        ]

        if all(marker in criteria_lower for marker in task_markers):
            return self._validate_engineering_tasks(answer)

        if all(marker in criteria_lower for marker in feature_markers):
            return self._validate_features(answer)

        if all(marker in criteria_lower for marker in user_story_markers):
            return self._validate_user_stories(answer)

        return (
            True,
            "No specialized deterministic structure check was required."
        )

    def evaluate(self, initial_prompt):
        # This method manages interactions between agents to achieve a solution.
        client = OpenAI(
            base_url=OPENAI_BASE_URL,
            api_key=self.openai_api_key
        )

        prompt_to_evaluate = initial_prompt
        response_from_worker = ""
        evaluation = "No evaluation completed."

        for i in range(self.max_interactions):  # TODO: 2 - Set loop to iterate up to the maximum number of interactions:
            print(f"\n--- Interaction {i + 1} ---")

            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")

            # TODO: 3 - Obtain a response from the worker agent
            response_from_worker = self.worker_agent.respond(
                prompt_to_evaluate
            )

            print(f"Worker Agent Response:\n{response_from_worker}")

            print(" Step 2: Evaluator agent judges the response")

            deterministic_passed, deterministic_reason = (
                self._run_deterministic_validation(
                    response_from_worker
                )
            )

            task_start_pattern = re.compile(
                r"(?mi)^\s*(?:\*\*)?Task ID:(?:\*\*)?\s*.+$"
            )
            original_task_count = len(
                task_start_pattern.findall(initial_prompt)
            )
            response_task_count = len(
                task_start_pattern.findall(response_from_worker)
            )

            if (
                deterministic_passed
                and original_task_count >= 2
                and response_task_count < original_task_count
            ):
                deterministic_passed = False
                deterministic_reason = (
                    "The corrected answer contains fewer engineering tasks "
                    f"than the original input. Expected at least "
                    f"{original_task_count} tasks, but found "
                    f"{response_task_count}. Preserve all valid tasks from "
                    "the original input."
                )

            print(
                "Deterministic Structure Validation:\n"
                f"{'PASS' if deterministic_passed else 'FAIL'}: "
                f"{deterministic_reason}"
            )

            eval_prompt = (
                "Evaluate the following answer strictly against the provided "
                "evaluation criteria.\n\n"
                "Answer to evaluate:\n"
                f"{response_from_worker}\n\n"
                "Evaluation criteria:\n"
                f"{self.evaluation_criteria}\n\n"  # TODO: 4 - Insert evaluation criteria here
                "Respond with Yes only if the answer fully satisfies every "
                "evaluation criterion. Otherwise, respond with No. Explain "
                "the reason after Yes or No. Verify every required field, "
                "field name, field value, ordering requirement, and content "
                "requirement before answering Yes. Instructions, templates, "
                "examples, placeholders, acknowledgments, or descriptions "
                "of the expected output do not satisfy criteria that require "
                "completed content."
            )

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[  # TODO: 5 - Define the message structure sent to the LLM for evaluation (use temperature=0)
                    {
                        "role": "system",
                        "content": self.persona
                    },
                    {
                        "role": "user",
                        "content": eval_prompt
                    }
                ],
                temperature=0
            )

            llm_evaluation = (
                response.choices[0].message.content.strip()
            )

            if not deterministic_passed:
                evaluation = (
                    "No. Deterministic validation failed: "
                    f"{deterministic_reason} "
                    "The evaluator cannot accept an answer that fails the "
                    "required structural checks."
                )
            elif deterministic_reason != (
                "No specialized deterministic structure check was required."
            ):
                evaluation = (
                    "Yes. The answer passed the specialized deterministic "
                    "validation required for this output structure. "
                    f"{deterministic_reason}"
                )
            else:
                evaluation = llm_evaluation

            print(f"Evaluator Agent Evaluation:\n{evaluation}")

            print(" Step 3: Check if evaluation is positive")

            if (
                deterministic_passed
                and evaluation.lower().startswith("yes")
            ):
                print("✅ Final solution accepted.")
                break

            print(
                " Step 4: Generate instructions to correct the response"
            )

            instruction_prompt = (
                "Provide precise correction instructions for the worker "
                "agent.\n\n"
                "Required evaluation criteria:\n"
                f"{self.evaluation_criteria}\n\n"
                "Answer that failed evaluation:\n"
                f"{response_from_worker}\n\n"
                "Reasons why the answer is incorrect:\n"
                f"{evaluation}\n\n"
                "Deterministic validation result:\n"
                f"{deterministic_reason}\n\n"
                "The correction instructions must follow the provided "
                "evaluation criteria exactly. Do not introduce different "
                "requirements, field names, output structures, or content "
                "requirements. Identify every unmet criterion. Require "
                "actual completed content instead of instructions, "
                "templates, examples, placeholders, acknowledgments, or "
                "descriptions of the expected format."
            )

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[  # TODO: 6 - Define the message structure sent to the LLM to generate correction instructions (use temperature=0)
                    {
                        "role": "system",
                        "content": self.persona
                    },
                    {
                        "role": "user",
                        "content": instruction_prompt
                    }
                ],
                temperature=0
            )

            instructions = (
                response.choices[0].message.content.strip()
            )

            print(f"Instructions to fix:\n{instructions}")

            print(
                " Step 5: Send feedback to worker agent for refinement"
            )

            prompt_to_evaluate = (
                "Correct the previous answer using only the original prompt, "
                "the required evaluation criteria, and the correction "
                "instructions below.\n\n"
                "Original prompt:\n"
                f"{initial_prompt}\n\n"
                "Previous answer:\n"
                f"{response_from_worker}\n\n"
                "Required evaluation criteria:\n"
                f"{self.evaluation_criteria}\n\n"
                "Deterministic validation failure:\n"
                f"{deterministic_reason}\n\n"
                "Correction instructions:\n"
                f"{instructions}\n\n"
                "Return the corrected and completed answer itself. Preserve "
                "all valid completed content from the original prompt and "
                "previous answer. Add or correct all content needed to "
                "satisfy every criterion. Do not discard valid user stories, "
                "features, or tasks. Do not return instructions, a template, "
                "an example, a placeholder, an acknowledgment, or an "
                "explanation of the expected format."
            )

        return {
            # TODO: 7 - Return a dictionary containing the final response, evaluation, and number of iterations
            "final_response": response_from_worker,
            "evaluation": evaluation,
            "iterations": i + 1
        }


class RoutingAgent:

    def __init__(self, openai_api_key, agents):
        # Initialize the agent with given attributes
        self.openai_api_key = openai_api_key

        # TODO: 1 - Define an attribute to hold the agents, call it agents
        self.agents = agents

    def get_embedding(self, text):
        client = OpenAI(
            base_url=OPENAI_BASE_URL,
            api_key=self.openai_api_key
        )

        # TODO: 2 - Write code to calculate the embedding of the text using the text-embedding-3-large model
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )

        # Extract and return the embedding vector from the response
        embedding = response.data[0].embedding

        return embedding

    # TODO: 3 - Define a method to route user prompts to the appropriate agent
    def route(self, user_input):
        # TODO: 4 - Compute the embedding of the user input prompt
        input_emb = self.get_embedding(user_input)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            # TODO: 5 - Compute the embedding of the agent description
            agent_emb = self.get_embedding(agent["description"])

            if agent_emb is None:
                continue

            denominator = (
                np.linalg.norm(input_emb)
                * np.linalg.norm(agent_emb)
            )

            if denominator == 0:
                similarity = 0.0
            else:
                similarity = float(
                    np.dot(input_emb, agent_emb) / denominator
                )

            print(similarity)

            # TODO: 6 - Add logic to select the best agent based on the similarity score between the user prompt and the agent descriptions
            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(
            f"[Router] Best agent: {best_agent['name']} "
            f"(score={best_score:.3f})"
        )

        selected_function = best_agent.get("func")

        if not callable(selected_function):
            return (
                f"Sorry, the selected agent '{best_agent['name']}' "
                "does not have a callable function."
            )

        return selected_function(user_input)


class ActionPlanningAgent:

    def __init__(self, openai_api_key, knowledge):
        # TODO: 1 - Initialize the agent attributes here
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def extract_steps_from_prompt(self, prompt):

        # TODO: 2 - Instantiate the OpenAI client using the provided API key
        client = OpenAI(
            base_url=OPENAI_BASE_URL,
            api_key=self.openai_api_key
        )

        # TODO: 3 - Call the OpenAI API to get a response from the "gpt-3.5-turbo" model.
        # Provide the following system prompt along with the user's prompt:
        # "You are an action planning agent. Using your knowledge, you extract from the user prompt the steps requested to complete the action the user is asking for. You return the steps as a list. Only return the steps in your knowledge. Forget any previous context. This is your knowledge: {pass the knowledge here}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an action planning agent. Using your "
                        "knowledge, you extract from the user prompt the "
                        "steps requested to complete the action the user is "
                        "asking for. You return the steps as a list. Only "
                        "return the steps in your knowledge. Forget any "
                        "previous context. This is your knowledge: "
                        f"{self.knowledge}"
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        # TODO: 4 - Extract the response text from the OpenAI API response
        response_text = response.choices[0].message.content

        # TODO: 5 - Clean and format the extracted steps by removing empty lines and unwanted text
        steps = response_text.split("\n")

        steps = [
            re.sub(
                r"^\s*(?:[-*]|\d+[.)])\s*",
                "",
                step
            ).strip()
            for step in steps
            if step.strip()
        ]

        return steps