# file q2.py
print("==[Q2:START]==")
import re

text = 'June 24, August 9, Dec 12, Will not Work, 23. I am 23, Zepelin 99, january 03'

# Lets use a regular expression to match a few date strings.

# CHANGE THE REGEX IN THE FOLLOWING LINE:
regex = re.compile(r'(June \d+|August \d+|Dec \d+|january \d+)') # the prefix "r'" is needed to avoid changing "\" into a string command

matches = regex.findall(text)

for match in matches:
    print(f'Matched string: {match}')

print("==[Q2:END]==")
