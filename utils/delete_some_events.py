import os
import shutil

def delete_some_events(img_dir):

    img_list = os.listdir(img_dir)

    for img_name in img_list:

        # 删除0-70与大于100的
        if int(img_name.split('_')[0])<70 or int(img_name.split('_')[0])>100:

            os.remove(os.path.join(img_dir,img_name))
            print(img_name)

delete_some_events('../datasets/train/data')