import secrets
import time



l = []
timeout = 0
flag = 0


def giveSecret():
    global timeout
    sc = secrets.token_hex(16)
    l.append(sc)
    timeout = time.time() + 60 * .5
    return  sc



def checkSecret(secret):
    global flag
    if time.time() > timeout:
        l.clear()
        return 'cannot reset your password - session timeout'
    else :
        for scc in l :
            if scc == secret:
                flag = 1
        if flag == 1 :
            flag = 0
            return  'Authenticated !!!'
        else :
            return  'Unknown Trace !!!'
