import json

from utils.utils import extract_json_string

response = '''{}
{
    "cv_passed_seeker_ids": []
}'''

a = extract_json_string(response)
print(a)

res_dict = json.loads(a)
print(res_dict)
b = list(map(int, res_dict["cv_passed_seeker_ids"]))
print(b)