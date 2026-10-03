import zipfile  

# 指定路径  
zip_path = "/public/home/cookie/anaconda3/envs/ocsort_env/OC_SORT/datasets/MOT-17/MOT17.zip"  
extract_dir = "/public/home/cookie/anaconda3/envs/ocsort_env/OC_SORT/datasets/MOT-17"  

# 执行解压  
with zipfile.ZipFile(zip_path, 'r') as zip_ref:  
    zip_ref.extractall(extract_dir)  
print(f"解压完成，文件保存在 {extract_dir}") 