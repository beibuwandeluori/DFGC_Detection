# Extract faces from videos

## Usage
- our default video_root_path is "../data_structure/Celeb-DF-v2"
- our default image_root_path is "../data_structure/Celeb-DF-v2-face"
1. Extrace Celeb-real faces  from Celeb-real videos, Celeb-synthesis faces from Celeb-synthesis videos"

```
python extract_video_celeb_df_v2.py --gpu_id 0 --video_root_path input_video_root_path --image_root_path output_image_root_path
```

2. Extrace YouTube-real faces from YouTube-real videos

```
python extract_video_celeb_df_v2_yotube.py --gpu_id 0 --video_root_path input_video_root_path --image_root_path output_image_root_path
```



3. Extract landmarks of face (resize to 256x256) and save as json files

```
python generate_landmarks_dlib_celeb_df_v2.py --image_root_path save_image_root_path
```

 

# Environment

```
facenet-pytorch==2.5.0
torch==1.3.1
dlib==19.21.1
```



