from .base import Agent

class CoderAgent(Agent):
    def process(self, task: str, context: str = "") -> str:
        prompt = self._build_prompt(task, context)
        prompt += """
Please provide the implementation. 
Format your response by specifying the filename for each file using the following format:
#### [FILE]: [path/to]/[filename].[ext] (make sure path and filename are correct and meaningful)
```[language]
# code here
```
"""
        return self.llm_client.generate_response(prompt)
