#!/usr/bin/env python3
A=print
import sys as C,os
from base64 import b64decode as F
G={A:A for A in['requests','urllib3','pandas']}
B=[]
for D in G.keys():
	try:__import__(D)
	except:B.append(D)
if B:A(f"pip install {" ".join(B)}");C.exit(1)
import requests as H,urllib3 as E,pandas as I
from urllib.parse import urlparse as J,parse_qs as K
E.disable_warnings(E.exceptions.InsecureRequestWarning)
def L(f):
	B='Predict'
	try:
		A=I.read_csv(f);C=['ID',B]
		if not all(B in A.columns for B in C):return 0,0,f"M:{[B for B in C if B not in A.columns]}"
		D=A[B][~A[B].isin([0,1])].unique()
		if len(D)>0:return 0,0,f"I:{D}"
		return 1,A,0
	except FileNotFoundError:return 0,0,f"N:{f}"
	except:return 0,0,'E'
def M(f):
	D='score'
	try:
		with open(f,'rb')as E:A=H.post(F(b'aHR0cHM6Ly9zdGF0NDAxMS1wYXJ0MS5zdGEuY3Voay5lZHUuaGsvdXBsb2Fk').decode(),files={'submission':E},data={'group_id':12507},allow_redirects=0,verify=0,timeout=30)
		if A.status_code==302:
			B=A.headers.get('Location')
			if not B:return 0,0,'NR'
			C=K(J(B).query)
			if D in C:
				try:return 1,float(C[D][0]),0
				except:return 0,0,'SE'
			return 0,0,'NS'
		return 0,0,f"S:{A.status_code}"
	except:return 0,0,'F'
def B():
	if len(C.argv)!=2:A('Usage: python u.py <csv>');return 2
	B=C.argv[1]
	if not os.path.exists(B):A(f"N:{B}");return 2
	E,H,D=L(B)
	if not E:A(f"E:{D}");return 3
	F,G,D=M(B)
	if F:A(f"F1:{G:.6f}");return 0
	A(f"E:{D}");return 4 if'NS'not in str(D)else 5
if __name__=='__main__':C.exit(B())