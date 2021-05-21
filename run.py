from flask import Flask, render_template, request 	
import base64, re, io
from PIL import Image
import base64
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(784, 520)
        self.l2 = nn.Linear(520, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 10)

    def forward(self, x):
        # Flatten the data (n, 1, 28, 28)-> (n, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)

# Need to open the pickled list object into read mode
 
model_pickle_path = 'model.pkl'
model_unpickle = open(model_pickle_path, 'rb')
 
# load the unpickle object into a variable
clf = pickle.load(model_unpickle)
		
app = Flask(__name__)

lastresult = 5

@app.route('/')
def main():
	with open('Stats', 'r') as f:
		string = f.readlines()[0]
		tot, cor = string.split()	
	tot = float(tot)
	cor = float(cor)
	return render_template('main.html', acc = (cor/tot))

@app.route('/recieve', methods = ['POST', 'GET'])
def recieve():
	image_b64 = request.values['imageBase64']
	image_data = base64.b64decode(re.sub('^data:image/png;base64,', '', image_b64))	
	image_PIL = Image.open(io.BytesIO(image_data))
	image_np = np.array(image_PIL)
	gsarr = []
	x = 0
	y = 0
	for i in range(28):
		for j in range(28):
			avg = 0
			for t in range(20):
				for k in range(20):
					avg += image_np[x+t][y+k][3]
			avg /= 400
			gsarr.append(avg)
			y += 20
		x += 20
		y = 0

	gsarr = np.array([gsarr])
	gsarr = torch.from_numpy(gsarr).type(torch.FloatTensor)
	# d = np.array(gsarr)
	# d.shape = (28, 28)
	# plt.imshow(255-d, cmap='gray')
	# plt.show()		
	global lastresult	
	lastresult = clf(gsarr).data.max(1, keepdim=True)[1][0][0].item()
	# print(lastresult)
	return "great"

@app.route('/update', methods = ['POST', 'GET'])
def update():
	data = request.get_json(force = True)
	with open('Stats', 'r') as f:
		string = f.readlines()[0]
		tot, cor = string.split()	
	
	with open('Stats', 'w') as f:	
		if data == 1:
			f.writelines([str(int(tot)+1), " ", str(int(cor)+1)])
		else:
			f.writelines([str(int(tot)+1), " ", cor])
	return "great"

@app.route('/result')
def result():
	return render_template('result.html', result = lastresult)

app.run()