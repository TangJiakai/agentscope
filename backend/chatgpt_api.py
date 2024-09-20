import openai
import concurrent.futures
from time import sleep
import json
import copy
from prompt_template import GPT_evaluate_prompt, GPT_Evaluate_And_Improve_Prompt_choice, GPT_Evaluate_And_Improve_Prompt_text, score_response_format, choice_rewritten_format, text_rewirtten_format, choice_prompt, text_prompt, choice_format, text_format


def chatgpt(data_):
    client = openai.OpenAI(api_key='XXX',
                            base_url='XXX')
    def api(data):
        completion = client.chat.completions.create(
                model='gpt-4o-2024-08-06',
                messages=[
                    {"role": "user", "content": data['prompt']}
                ],
                response_format=data['response_format'],
                temperature=1
        )
        return completion.choices[0].message.content.strip()

    while True:
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                res = list(executor.map(api, data_))
            print('Done!')
            return res
        except KeyboardInterrupt:
            exit(1)
        except Exception as e:
            print(f"Catching Exception {e}, Retrying...")
            sleep(1)
            continue


def rewritten_responses(data):
    data_submit = []
    for d in data:
        if d['selection_num']:
            enum = [str(i) for i in range(d['selection_num'])]
            choice_format["json_schema"]["schema"]["properties"]["rewritten_response"]["enum"] = enum
            json_data = {"prompt": choice_prompt.format(prompt=d["prompt"], completion=d["completion"]), "response_format": choice_format}
            data_submit.append(copy.deepcopy(json_data))
        else:
            json_data = {"prompt": text_prompt.format(prompt=d["prompt"], completion=d["completion"]), "response_format": text_format}
            data_submit.append(copy.deepcopy(json_data))
    
    res_ = chatgpt(data_submit)
    res = [json.loads(r)["rewritten_response"] for r in res_]
    return res


def rate_responses(data):

    res = chatgpt([{"prompt": GPT_evaluate_prompt.format(prompt=d["prompt"], completion=d["completion"]),
                    "response_format": score_response_format} for d in data])
    res = [int(json.loads(r)["rating"]) for r in res]
    return res


if __name__ == "__main__":
    data = [
        {
        "msg_id": None,
        "agent_id": "67665cbd1a1d4856b6502cff31498e54",
        "name": "Ethan Rodriguez",
        "agent_type": "SeekerAgent",
        "prompt": "## Conversation History\nuser: [INSTRUCTION]\n\nYou are a job seeker.\nNow you need to search for jobs and consider applying for these positions. The higher the intensity of your job search, the more positions you will search for, and the greater your chances of finding a job that suits you. \nBased on your current personal situation, please choose an integer to maximize the total discounted utility to the present value across all periods. \n\n[\\INSTRUCTION]\n\n[PROFILE]\n\n        Name: Ethan Rodriguez\n        CV: {'Education': \"Bachelor's Degree in Software Engineering from University of Texas\", 'Work Experience': [{'Company': 'Digital Solutions', 'Position': 'Frontend Developer', 'Time': '2016-2019'}, {'Company': 'AI Startups', 'Position': 'Full Stack Developer', 'Time': '2019-Present'}], 'Skills': ['JavaScript', 'React', 'Node.js', 'Artificial Intelligence', 'Database Management']}\n        Trait: {'Age': 30, 'Gender': 'Male', 'Patience': 'Average', 'Risk Tolerance': 'High', 'Positive Reciprocity': 'Above Average', 'Negative Reciprocity': 'Moderate', 'Altruism': 'Average', 'Trust': 'Above Average'}\n        Working Condition: unemployed\n        \n[\\PROFILE]\n\n\nPlease make a choice from the actions below (output the index, e.g., 0):\n\n    (0) 1\n\n    (1) 2\n\n    (2) 3\n\n    (3) 4\n\n    (4) 5\n\nAnswer:\n",
        "completion": "1",
        "rewritten_response": None,
        "rating": None,
        "selection_num": 5
        },
    ]
    print(rewritten_responses(data))
    print(rate_responses(data))