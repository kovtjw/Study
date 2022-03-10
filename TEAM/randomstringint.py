import string
import random

_length = 5

string_pool = string.ascii_letters + string.digits

result = ''
for i in range(_length):
    result += random.choice(string_pool)
print(result)