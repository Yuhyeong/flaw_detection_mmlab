import os
import shutil


def mov_label(all_events_dir_path, out_labels_path):
    if not os.path.exists(out_labels_path):
        os.mkdir(out_labels_path)

    events_list = os.listdir(all_events_dir_path)

    for eve_name in events_list:

        single_eve_path = os.path.join(all_events_dir_path, eve_name)

        eve_labels_list = os.listdir(single_eve_path)

        print(eve_name)

        for single_label in eve_labels_list:
            single_eve_label_path = os.path.join(single_eve_path, single_label)

            single_out_label_path = os.path.join(out_labels_path, eve_name + single_label[5:])

            shutil.move(single_eve_label_path, single_out_label_path)


def correct_label(all_events_dir_path, out_labels_path):
    events_list = os.listdir(all_events_dir_path)

    for eve_name in events_list:

        single_eve_path = os.path.join(all_events_dir_path, eve_name)

        eve_labels_list = os.listdir(single_eve_path)

        # print(eve_name)

        for single_label in eve_labels_list:
            single_eve_label_path = os.path.join(single_eve_path, single_label)

            f = open(single_eve_label_path, 'r', encoding='utf-8')
            txt = f.read()
            flag = txt.find('d')
            f.close()

            if flag != -1:
                print(single_eve_label_path)
                txt = txt.replace('d', '3')

            f = open(single_eve_label_path, 'w', encoding='utf-8')
            f.write(txt)
            f.close()


mov_label('D:\\BaiduNetdiskDownload\\denoise_output_gaussian_label', 'D:\\BaiduNetdiskDownload\\label')
