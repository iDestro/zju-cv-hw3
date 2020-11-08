import cv2
import numpy as np


def visualize(images, predict_images, config):
    cnt = 0
    img_cnt = 0
    factor = 25 if config.target_type =='multi' else 250
    while cnt+100 <= len(images):
        temp_cnt = cnt
        temp_origin = None
        for i in range(10):
            temp_0 = None
            for j in range(10):
                if temp_0 is None:
                    temp_0 = images[cnt]
                else:
                    temp_0 = np.concatenate([temp_0, images[cnt]], axis=1)
                cnt += 1
            if temp_origin is None:
                temp_origin = temp_0
            else:
                temp_origin = np.concatenate([temp_origin, temp_0], axis=0)

        # 存放预测图片
        temp_predict = None
        cnt = temp_cnt
        for i in range(10):
            temp_0 = None
            for j in range(10):
                if temp_0 is None:
                    temp_0 = predict_images[cnt]*factor
                else:
                    temp_0 = np.concatenate([temp_0, predict_images[cnt]*factor], axis=1)
                cnt += 1
            if temp_predict is None:
                temp_predict = temp_0
            else:
                temp_predict = np.concatenate([temp_predict, temp_0], axis=0)
        img_cnt += 1
        res = np.concatenate([temp_origin, temp_predict], axis=1)
        path = './result/multi/' if config.target_type =='multi' else './result/binary/'
        print(img_cnt)
        cv2.imwrite(path + str(img_cnt) + '.jpg', res)