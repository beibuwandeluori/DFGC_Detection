## Usage

Split the train and val dataset. Save the video names. Here we've given txt file, so you can directly use it.

```
python generate_txt.py --video_root_path video_root_path
```



### Train different baseline models

```
python train_binary_resnet18.py --gpu_id 0 --root_path image_root_path
```



