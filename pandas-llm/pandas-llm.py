import pandas as pd
import datetime
import numpy as np
import time
import openai
import os
from sandbox import Sandbox
import re
import json



class PandasLLM(pd.DataFrame):

    model = "gpt-3.5-turbo"
    temperature = 0.2
    code_blocks = [r'```python(.*?)```',r'```(.*?)```']
    privacy = True

    def __init__(self, data=None, config=None, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        
        self.config = config or {}

        # Set up OpenAI API key from the environment or the config
        self.openai_api_key = os.environ.get("OPENAI_KEY") or self.config.get("openai_api_key", "your_openai_api_key")

    def buildPromptForRole(self):
        prompt_role = f"""
I want you to act as a data scientist and Python coder. I want you code for me. 
I have a dataset of {len(self)} rows and {len(self.columns)} columns.
Columns and their type are the following:
        """

        for col in self.columns:
            col_type = self.dtypes[col]
            prompt_role += f"{col} ({col_type})\n"


        return prompt_role

    def buildPromptForProblemSolving(self, request):

#         prompt_problem = f"""
# Given a DataFrame named 'df', write a Python code snippet to address the following request:
# {request}

# Please adhere to these guidelines while crafting the code:
# 1. When comparing or search strings, always use lower case letters and ignore case sensitivity and apply a "contains" search.
# 2. If a request involves searching for a string without specifying a column (i.e. "search for Milk"), search in both category columns and product name columns.
# 3. if not requested explicitely, the result will show only the column indicated in the request.

# Make sure the answer is a single line of code without explanations, comments, or additional details. 
# if a solution in a single line of code is not possible, you are allowed to write multi line solution, even functions but the end of the code must be an assignment to the variable 'result'.
# Assign the resulting code to the variable 'result'.

        prompt_problem = f"""

Given a DataFrame named 'df', write a Python code snippet that addresses the following request:
{request}

While crafting the code, please follow these guidelines:
1. When comparing or searching for strings, use lower case letters, ignore case sensitivity, and apply a "contains" search.
2. If a request involves searching for a string without specifying a column (e.g., "search for Milk"), search in both category columns and product name columns.
3. Unless explicitly requested, the result should only show the column(s) indicated in the request.

Ensure that the answer is a single line of code without explanations, comments, or additional details. 
If a single line solution is not possible, multiline solutions or functions are acceptable, but the code must end with an assignment to the variable 'result'. 
Assign the resulting code to the variable 'result'.
Avoid importing any additional libraries than pandas and numpy.
        """

        return prompt_problem

    def extractPythonCode(self, text: str, regexp: str) -> str:
        # Define the regular expression pattern for the Python code block
        pattern = regexp
        
        # Search for the pattern in the input text
        match = re.search(pattern, text, re.DOTALL)
        
        # If a match is found, return the extracted code (without the markers)
        if match:
            return match.group(1).strip()
        
        # If no match is found, return an empty string
        return ""

    def variable_to_string(self, variable):
        if variable is None: return None
        try:

            if isinstance(variable, pd.Series):
                # convert to dataframe
                variable = variable.to_frame()

            if isinstance(variable, pd.DataFrame):
                variable = variable.drop_duplicates()
                if len(variable) == 0: return None
                return str(variable)

            elif isinstance(variable, np.ndarray):
                if len(variable) == 0: return None
                return  np.array2string(variable)
            else:
                # Convert the variable to a string
                return str(variable)
        except Exception as e:
            return str(variable)
        

    def save(self,name,value):
        try:
            with open(name, 'w') as file:
                file.write(value)
        except Exception as e:
            print(e)

    def execInSandbox(self, df, generated_code:str):

        # Create a Sandbox instance and allow pandas to be imported
        sandbox = Sandbox()
        sandbox.allow_import("pandas")
        sandbox.allow_import("numpy")
        # sandbox.allow_import("datetime")

        # Define the initial code to set up the DataFrame
        # dict = df.to_dict()
        initial_code = f"""
import pandas as pd
import datetime
from pandas import Timestamp
import numpy as np

        """

        # Combine the initial code and the generated code
        full_code = initial_code + "\n" + generated_code

        self.save("temp/prompt_code.py",full_code)
        # Execute the combined code in the Sandbox
        sandbox_result = sandbox.execute(full_code, {"df":df})

        # Get the result from the local_vars dictionary
        result = sandbox_result.get("result")
        return result



    def prompt(self, request: str):
        # Set up OpenAI API key
        openai.api_key = self.openai_api_key

        messages=[
                {"role": "system", 
                "content": self.buildPromptForRole()},
                {"role": "user", 
                "content": self.buildPromptForProblemSolving(request)
                }
            ]

        response = None
        for times in range(0,3):
            try:
                response = openai.ChatCompletion.create(
                model=self.model,
                temperature=self.temperature,
                messages = messages
                )
                break;
            except Exception as e:
                print(f"error {e}")
                continue

        if response is None:
            return "Please try later"

        self.save("temp/prompt_cmd.json",json.dumps(messages, indent=4))

        generated_code = response.choices[0].message.content
        if generated_code == "" or generated_code is None:
            return "Please try again with a different question"
        
        results=[]
        for regexp in self.code_blocks:
            cleaned_code = self.extractPythonCode(generated_code,regexp)
            if cleaned_code == "" or cleaned_code is None:
                continue
            results.append(cleaned_code)
        results.append(generated_code)

        if len(results) == 0:
            return "Please try again with a different question"


        for cleaned_code in results:

            result = None
            try:
                result = self.execInSandbox(self, cleaned_code)
            except Exception as e:
                print(f"error {e}")
                try:
                    expression = re.sub(r"^\s*result\s*=", "", cleaned_code).strip()
                    result = eval(expression, {'df': self, 'pd': pd, 'np': np, 'datetime': datetime, 'result': result})
                except Exception as e:
                    print(f"error {e}")
                    pass

            if result is not None and str(result) != "":
                break

        formatted_result = self.variable_to_string(result)
        # formatted_result = str(result)

        # check if the result is empty
        if formatted_result is None or formatted_result == "" or formatted_result.strip() == "" or len(formatted_result) == 0:
            return "Please try again with a different question" 

        self.save("temp/prompt_result.json",formatted_result)

        if self.privacy == True:
            return formatted_result

        messages=[
                {"role": "system", 
                "content": """
I want you to act as a data interpreter. 
you will receive data in different formats and you will explain the data in different formats
                            """},
                {"role": "user", 
                "content": 

f"""
The user posed this question:
'{request}'

The answer is provided as a payload with the following data:
{formatted_result}

Please present the data in a human-readable format, using lists, tables, or other straightforward visualizations. 
Refrain from using JSON or complex formats in the presentation. 
Ensure that the output is comprehensible to a non-technical audience. 
After displaying the data, provide an explanation of its contents but only if it is not obvious from the presentation.
certain answers apparently are no making sense, but we will accept it, especially if contains code or identifiers.
"""
                }
            ]

        
        self.save("temp/prompt_desc.json",json.dumps(messages, indent=4))

        response = None
        for retry in range(3):
            try:
                response = openai.ChatCompletion.create(
                model=self.model,
                temperature=self.temperature,
                messages = messages
                )
                break
            except Exception as e:
                print(f"error {e}")
                print(f"retry {retry}")
                time.sleep(5)

        if response is None:
            return result
        
        result = response.choices[0].message.content

        return result

