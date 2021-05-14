# Generate adversarial examples (Attacking images)

# Training baseline models for attack
* Before Attacking images, you need to train several models using Celeb-DF-v2 data:
Xception(official baseline, the weight was provided by the official),
ResNet34,ResNet50,VGG16,VGG16_BN,VGG19 and VGG19_BN. The training code provided in "../train_baseline_model/*"
* Or you can download the pretrained weights in 百度网盘（pan.baidu.com）:

链接(download link)：https://pan.baidu.com/s/1GhBdFEwWdBUN3rz_lsdNqg 
提取码(Extracted code)：1234 

After downloaded the weights, you can unzip it and move those weigths into "weights" 
## Usage

### parameter
```
python attack_ensemble_example{i|i={1,2,3,4,5}}.py
    --input_path  input_video_path contains every frame image from video
    --use_mask  if use mask to attack
    --frames  each video frame nums
    --batch_size model batch_size 
    --steps  Adversarial algorithm parameter: the max Iterations
    --max_norm   Adversarial algorithm parameter: the max norm of image
    --div_prob probability of Input_diversity
    --gpu_id 
```
* 1.Xception(official baseline),ResNet34,VGG16,VGG19_BN - TPGD without_mask
```
python attack_ensemble_example1.py --gpu_id 0 --input_path your_face_root/Celeb-synthesis
```
the output adv images can seen in your_face_root/Celeb-synthesis_adv1

* 2.Xception(official baseline),ResNet34,VGG16_BN,VGG19 - TPGD use_mask
```
python attack_ensemble_example2.py --gpu_id 0  --input_path your_face_root/Celeb-synthesis --use_mask
```
the output adv images can seen in your_face_root/Celeb-synthesis_adv2

* 3.Xception(official baseline),ResNet34,ResNet50,VGG19_BN - MI-FGSM use_mask
```
python attack_ensemble_example3.py --gpu_id 0  --input_path your_face_root/Celeb-synthesis --use_mask
```
the output adv images can seen in your_face_root/Celeb-synthesis_adv3

* 4.Xception(official baseline),ResNet34,VGG16_BN,VGG19 - MI-FGSM use_mask
```
python attack_ensemble_example4.py --gpu_id 0  --input_path your_face_root/Celeb-synthesis --use_mask
```
the output adv images can seen in your_face_root/Celeb-synthesis_adv4

* 5.Xception(official baseline),ResNet34,Res50,VGG19_BN - TPGD use_mask
```
python attack_ensemble_example5.py --gpu_id 0  --input_path your_face_root/Celeb-synthesis --use_mask
```
the output adv images can seen in your_face_root/Celeb-synthesis_adv5
 
# Environment

```
opencv-python==4.2.0
pillow==7.1.2
torch==1.3.1
dlib==19.21.1
```



