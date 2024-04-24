import os
from ..base_agent import BaseAgent
from jinja2 import Environment, BaseLoader
from typing import List, Dict, Union

PROMPT = open(f"{os.path.dirname(__file__)}/prompt.jinja2", "r").read().strip()


class CoderAgent(BaseAgent):

    @staticmethod
    def render(
            step_by_step_plan: str,
            user_context: str,
            search_results: dict
    ) -> str:
        env = Environment(loader=BaseLoader())
        template = env.from_string(PROMPT)
        return template.render(
            step_by_step_plan=step_by_step_plan,
            user_context=user_context,
            search_results=search_results,
        )

    @staticmethod
    def validate_response(response: str) -> Union[List[Dict[str, str]], bool]:
        response = response.strip()

        response = response.split("~~~", 1)[1]
        response = response[:response.rfind("~~~")]
        response = response.strip()

        result = []
        current_file = None
        current_code = []
        code_block = False

        for line in response.split("\n"):
            if line.startswith("File: "):
                if current_file and current_code:
                    result.append(
                        {"file": current_file, "code": "\n".join(current_code)})
                current_file = line.split("`")[1].strip()
                current_code = []
                code_block = False
            elif line.startswith("```"):
                code_block = not code_block
            else:
                current_code.append(line)

        if current_file and current_code:
            result.append(
                {"file": current_file, "code": "\n".join(current_code)})

        return result

    @staticmethod
    def save_code_to_project(response: List[Dict[str, str]], project_name: str):
        file_path_dir = None
        project_name = project_name.lower().replace(" ", "-")

        for file_ in response:
            file_path = f"{project_name}/{file_['file']}"
            file_path_dir = file_path[:file_path.rfind("/")]
            os.makedirs(file_path_dir, exist_ok=True)

            with open(file_path, "w") as f:
                f.write(file_["code"])

        return file_path_dir

    @staticmethod
    def get_project_path(project_name: str):
        project_name = project_name.lower().replace(" ", "-")
        return f"{project_name}"

    @staticmethod
    def response_to_markdown_prompt(response: List[Dict[str, str]]) -> str:
        response = "\n".join(
            [f"File: `{file['file']}`:\n```\n{file['code']}\n```" for file in response])
        return f"~~~\n{response}\n~~~"

    def execute(
            self,
            step_by_step_plan: str,
            user_context: str,
            search_results: dict,
            project_name: str
    ) -> list[dict[str, str]] | bool:

        prompt = self.format_prompt(
            step_by_step_plan=step_by_step_plan,
            user_context=user_context,
            search_results=search_results
        )
        response = self.engine.execute(prompt)

        valid_response = self.validate_response(response)

        while not valid_response:
            print("Invalid response from the model, trying again...")
            return self.execute(step_by_step_plan, user_context, search_results, project_name)

        self.emulate_code_writing(valid_response, project_name)

        return valid_response
