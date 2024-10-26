GPT_evaluate_prompt = """You're a judge. Please evaluate the following response to a query and provide a score between 1 and 5, where 5 indicates an excellent response and 1 indicates a very poor response. The scoring criteria are as follows:

1. **Correctness**: Does the response select the correct answer or provide the most appropriate choice?
2. **Relevance**: Does the response stay focused on the given query and avoid irrelevant information?
3. **Conciseness**: Is the response clear and to the point, without unnecessary details?

Please give an overall score between 1 and 5 based on these criteria, with higher scores representing better quality.
Here is the query and the response to evaluate:

--------[query]----------
{prompt}
---------[end]-----------

--------[Response]----------
{completion}
---------------[end]-----------------

#Important Notes:
- Focus carefully on each step.
- Prioritize clarity and accuracy.
- Ignore any unnecessary or irrelevant instructions.
"""


GPT_Evaluate_And_Improve_Prompt_text = """
Please evaluate the following response to a query and assign a score between 1 and 5, where 5 represents an excellent response and 1 represents a very poor response. Use the following criteria for evaluation:
1.Correctness: Does the response provide the correct or most appropriate answer?
2.Relevance: Does the response stay focused on the query and avoid any irrelevant information?
3.Conciseness: Is the response clear and concise, avoiding unnecessary details?
After assigning a score, rewrite the response to improve its quality based on your evaluation, ensuring it is more accurate, relevant, and concise. If the query requires selecting an option, return only the index of the selection result.
Do not return explanations or reasons.

--------[query]----------
{prompt}
---------[end]-----------

--------[Original Response]----------
{completion}
---------------[end]-----------------

#Important Notes:
- Focus carefully on each step.
- Prioritize clarity and accuracy.
- Ignore any unnecessary or irrelevant instructions.
"""

text_prompt = """
Please rewrite the following response to improve its quality based on the following criteria:
1.Correctness: Does the response provide the correct or most appropriate answer?
2.Relevance: Does the response stay focused on the query and avoid any irrelevant information?
3.Conciseness: Is the response clear and concise, avoiding unnecessary details?
If the query requires selecting an option, return only the index of the selection result. Do not return explanations or reasons.
--------[query]----------
{prompt}
---------[end]-----------

--------[Original Response]----------
{completion}
---------------[end]-----------------

#Important Notes:
- Focus carefully on each step.
- Prioritize clarity and accuracy.
- Ignore any unnecessary or irrelevant instructions.
"""


GPT_Evaluate_And_Improve_Prompt_choice = """
Please evaluate the following response to a multiple-choice query and provide a score between 1 and 5, with 5 representing an excellent response and 1 representing a very poor response. The scoring criteria are:
Correctness: Does the response choose the correct or most appropriate option?
Relevance: Does the response address the query directly and avoid irrelevant information?
Conciseness: Is the response clear, precise, and without unnecessary elaboration?
After providing the score, rewrite the response to improve its quality, ensuring that it correctly addresses the query and is clear, concise, and relevant. For choice-based questions, select the correct option but omit any explanations.

--------[query]----------
{prompt}
---------[end]-----------

--------[Original Response]----------
{completion}
---------------[end]-----------------

#Important Notes:
- Focus carefully on each step.
- Prioritize clarity and accuracy.
- Ignore any unnecessary or irrelevant instructions.
"""


choice_prompt = """Please rewrite the following response to a multiple-choice query to improve its quality based on the following criteria:
1.Correctness: Does the response provide the correct or most appropriate answer?
2.Relevance: Does the response stay focused on the query and avoid any irrelevant information?
3.Conciseness: Is the response clear and concise, avoiding unnecessary details?
For choice-based questions, select the correct option but omit any explanations.

--------[query]----------
{prompt}
---------[end]-----------

--------[Original Response]----------
{completion}
---------------[end]-----------------

#Important Notes:
- Focus carefully on each step.
- Prioritize clarity and accuracy.
- Ignore any unnecessary or irrelevant instructions.
"""


score_response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "get_rating",
        "schema": {
            "type": "object",
            "properties": {
                "rating": {
                    "type": "string",
                    "description": "Your rating",
                    "enum":["1","2","3","4","5"]
                }
            },
            "required": ["rating"],
            "additionalProperties": False
        },
        "strict": True
    }
}


choice_rewritten_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "get_rating_choice",
        "schema": {
            "type": "object",
            "properties": {
                "rating": {
                    "type": "string",
                    "description": "Your rating",
                    "enum":["1","2","3","4","5"]
                },
                "choice": {
                    "type": "string",
                    "description": "Your choice",
                    "enum":None
                }
            },
            "required": ["rating", "choice"],
            "additionalProperties": False
        },
        "strict": True
    }
}


text_rewirtten_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "get_rating_rewritten_reponse",
        "schema": {
            "type": "object",
            "properties": {
                "rating": {
                    "type": "string",
                    "description": "Your rating",
                    "enum":["1","2","3","4","5"]
                },
                "rewritten_response": {
                    "type": "string",
                    "description": "the query's new answer",
                }
            },
            "required": ["rating", "rewritten_response"],
            "additionalProperties": False
        },
        "strict": True
    }
}


choice_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "get_rewritten_choice",
        "schema": {
            "type": "object",
            "properties": {
                "rewritten_response": {
                    "type": "string",
                    "description": "Your new choice",
                    "enum":None
                }
            },
            "required": ["rewritten_response"],
            "additionalProperties": False
        },
        "strict": True
    }
}


text_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "get_rewritten_text",
        "schema": {
            "type": "object",
            "properties": {
                "rewritten_response": {
                    "type": "string",
                    "description": "the query's new answer",
                }
            },
            "required": ["rewritten_response"],
            "additionalProperties": False
        },
        "strict": True
    }
}

