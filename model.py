from fastai.vision import *
from fastai.metrics import accuracy
from PIL import Image

def pred(image):


	learn = load_learner(Path(''), 'Pokemon')


	image2 = open_image(image)
	           
	pred_class, pred_idx, pred_outputs = learn.predict(image2)

	return pred_class



