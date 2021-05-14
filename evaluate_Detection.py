#!/usr/bin/env python
from submission_Det import model as baseDet
from sklearn.metrics import roc_auc_score

img_dir = "./sample_imgs"
json_file = 'sample_meta.json'


# load detection model
print('loading baseline detection model...')
detModel = baseDet.Model(device_id=4)

print('Detecting images ...')
img_names, prediction = detModel.run(img_dir, json_file)
assert isinstance(prediction, list)
assert isinstance(img_names, list)

labels = [0]*5 + [1]*5
score = roc_auc_score(labels, prediction)
print('AUC score is %f' % score)

# 打包 zip ../submission_det.zip -r *
# zip ../submission_creat.zip -r *