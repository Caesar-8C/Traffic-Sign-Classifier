import tensorflow as tf
import numpy as np
import pickle
from scipy.special import softmax
import cv2
import glob

test_orig  = pickle.load(open( 'trafsignsDataset/test.p', 'rb'))
test = test_orig.copy()
test['features']  = (( test['features'].astype(np.int)-128)/128)

model = tf.keras.models.load_model('models/model3')

result = model.predict(test['features'])
result = softmax(result, axis=1)


tp = np.array([0 for _ in range(43)])
fp = np.array([0 for _ in range(43)])
fn = np.array([0 for _ in range(43)])

for i in range(result.shape[0]):
	s = sorted(result[i], reverse=True)
	print('===================================')
	print('i: ', i, ', label: ', test['labels'][i], end='')
	for j in range(5):
		guess = np.argwhere(result[i]==s[j])[0,0]
		if j == 0:
			if guess == test['labels'][i]:
				tp[guess] += 1
			else:
				fp[guess] += 1
				fn[test['labels'][i]] += 1
			print(', guess: ', str(guess))
		if guess < 10: guesstr = ' ' + str(guess)
		else: guesstr = str(guess)
		print(j+1, 'th guess is ' + guesstr + ', prob: ' + str(round(s[j], 2)))
	print('===================================')
	img = cv2.cvtColor(test_orig['features'][i], cv2.COLOR_RGB2BGR)
cv2.destroyAllWindows()
print('Test accuracy: ', round(np.sum(tp)/test['features'].shape[0]*100, 2), '%')
print('Precision and Recall for all classes:')
for i in range(43):
	P = tp[i]/(tp[i]+fp[i])
	R = tp[i]/(tp[i]+fn[i])
	print('Class ', i, ' P = ', round(P, 2), 'R = ', round(R, 2))