import openai
import concurrent.futures
from time import sleep


def chatgpt(data):
    client = openai.OpenAI(api_key='sk-nGk7efY9pqkFQngP174238D7Fd1e4775Bc96F40348376f2b',
                            base_url='https://api2.aigcbest.top/v1')
    def api(prompt):
        completion = client.chat.completions.create(
                model='gpt-3.5-turbo-0125',
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
        )
        return completion.choices[0].message.content.strip()

    while True:
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                res = list(executor.map(api, data))
            print('Done!')
            return res
        except KeyboardInterrupt:
            exit(1)
        except Exception as e:
            print(f"Catching Exception {e}, Retrying...")
            sleep(1)
            continue


def rewritten_responses(data):
    template = """Please now act as an excellent simulator. For the given query, you should
        answer the question reasonably and as realistically as possible, avoiding logical inconsistencies and errors that do not match reality.
        Please directly answer the question.
        
        Query: I am a female student studying liberal arts with a weaker foundation in mathematics and science, but I have excellent expressive and language comprehension abilities. Currently, I have several positions I can apply for, such as chef, programmer, and human resources. Which type of position should I apply for?
        Response: Human Resources.

        Query: {prompt}
        Response: """
    
    res = chatgpt([template.format(query=d["prompt"]) for d in data])
    return res


def rate_responses(data):
    template = """Please now act as an excellent simulator. 
    For the given query, you need to score the replies from 1 to 5 based on their reasonableness, where a lower score indicates lower reasonableness.
    Please directly answer the question.
    
    Query: I am a female student studying liberal arts with a weaker foundation in mathematics and science, but I have excellent expressive and language comprehension abilities. Currently, I have several positions I can apply for, such as chef, programmer, and human resources. Which type of position should I apply for?
    Response: Human Resources.
    Rating: 5

    Query: I am a female student studying liberal arts with a weaker foundation in mathematics and science, but I have excellent expressive and language comprehension abilities. Currently, I have several positions I can apply for, such as chef, programmer, and human resources. Which type of position should I apply for?
    Response: Programmer.
    Rating: 1
    
    Query: {prompt}
    Response: {completion}
    Rating: """

    res = chatgpt([template.format(query=d["prompt"], response=d["completion"]) for d in data])
    res = [int(r) for r in res]
    return res


if __name__ == "__main__":
    data = [
        {
            "prompt": "I am a female student studying liberal arts with a weaker foundation in mathematics and science, but I have excellent expressive and language comprehension abilities. Currently, I have several positions I can apply for, such as chef, programmer, and human resources. Which type of position should I apply for?",
            "completion": "Human Resources."
        },
    ]
    print(rewritten_responses(data))
    print(rate_responses(data))