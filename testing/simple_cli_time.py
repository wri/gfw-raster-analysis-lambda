import requests
import subprocess

url = r'https://0yvx7602sb.execute-api.us-east-1.amazonaws.com/dev/mill?lat=-2.19672724&lon=112.8515625'

r = requests.get(url)
resp = r.json()['result']

s3_output = str([x for x in resp.split() if 's3' in x][0])
print s3_output

for i in range(0,1000):

    cmd = ['aws', 's3', 'ls', s3_output]

    try:
        out = subprocess.check_output(cmd).split('\n')
        if len(out) == 9:
            break
    	print len(out)
    except:
        pass
