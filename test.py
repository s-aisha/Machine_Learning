import time
from hidden import giveSecret,checkSecret



# authentication at right timer

print('authentication at right timer')
secret = giveSecret()
print(secret)
print(checkSecret(secret))


# authentication at right timer

print('authentication after time out')

timeout = time.time() + 60 * .5
print(time.time())
print(timeout)
while True:
    if time.time() > timeout:
        print(checkSecret(secret))
        break