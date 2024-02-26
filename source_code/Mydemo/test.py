from matplotlib import pyplot
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
from tqdm import tqdm


def cal_entropy(prob):
    """
    calculate entropy of each defect
    """
    entropy = -1 * np.sum(prob * np.log2(prob))
    return entropy


def cal_entropy_one_image(img_path, model, times):
    img_path_list = [img_path for i in range(times)]
    results = model(img_path_list)
    entropy = []
    for re in results:
        cls_all = re.cls_all
        if len(cls_all) != 0:
            cls_all = np.array(cls_all.cpu())
            entropy_sum = 0
            for i in range(len(cls_all)):
                entropy_sum += cal_entropy(cls_all[i])
            entropy.append(entropy_sum / len(cls_all)) 
        
        #plot
        im_array = re.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        # plot img
        pyplot.imshow(im_array)
        pyplot.show()     
                
    entropy_mean = np.mean(np.array(entropy))
    
    return entropy_mean, results

# cluster the output bounding boxes
from sklearn.cluster import KMeans
def cluster_bounding_boxes(bounding_boxes:np.ndarray, n_clusters=3, confs=None, threshold=0.5):
    """_summary_
    Args:
        bounding_boxes (_type_): shape (boxes_num, 4)
        
    Return:
        total_variance should be weighted according to the numbers of corrosponding labels
    """
    selected_data = bounding_boxes[confs > threshold]
    
    if n_clusters == 0 or len(selected_data) == 0:
        return selected_data, None, 0, 0

    if len(selected_data) < n_clusters: 
        n_clusters = len(selected_data)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, max_iter=1000, n_init="auto").fit(selected_data)
    labels = kmeans.labels_

    variances = []
    cls_entropy = []
    for i in range(n_clusters):
        cluster_data = selected_data[labels == i]
        
        if len(cluster_data) == 0:
            variances.append(0)     
        else:
            variances.append(np.var(cluster_data, axis=0))
    
    weighted_variance = np.array(variances) * np.bincount(labels, minlength=n_clusters).reshape(-1, 1) / np.sum(np.bincount(labels))
    weighted_variance_sum = np.sum(np.mean(weighted_variance, axis=1))
  
    return selected_data, labels, variances, weighted_variance_sum

def uncertainty(dir, times, model, df, save_name, folder='test'):
    
    df = pd.DataFrame(columns=['img_name', 'objectness_uncertainty', 'weighted_variance_sum', 'weighted_entropy'])
    images = os.listdir(os.path.join(dir, folder, "images"))
    labels = os.listdir(os.path.join(dir, folder, "labels"))
    
    objectness_sum = []
    weighted_variance_total = []
    weighted_entropy_total = []
    for img, label in tqdm(zip(images, labels), total=len(images), desc="testing"):
        img_path = os.path.join(dir, folder, "images", img)
        label_path = os.path.join(dir, folder, "labels", label)
    
        results = model([img_path for i in range(times // 2)], verbose=False)
        results.extend(model([img_path for i in range(times - times // 2)], verbose=False))
        
        boundingboxes = []
        boxes = []
        conf = []
        cls_conf = []
        predict_cluster = []
        for re in results:
            conf.extend(re.boxes.conf.cpu())
            tmp = re.boxes.xywhn
            boxes.extend(re.boxes.cpu())
            boundingboxes.extend(tmp.cpu())
            cls_conf.extend(re.cls_all.cpu())
            predict_cluster.append(re.boxes.shape[0])
            
        #read txt file
        # cluster_num = len(np.loadtxt(label_path)) #using groundtruth
        
        #set cluster_num equal to the 
        #get most frequent number
        cluster_num = max(set(predict_cluster), key=predict_cluster.count)
        
        boundingboxes = np.array(boundingboxes)
        conf = np.array(conf)

        # if cluster_num == 0: #no label in image 
        #     continue
        
        selected_data, labels, variances, weighted_variance_sum = cluster_bounding_boxes(boundingboxes, n_clusters=cluster_num, confs=np.array(conf), threshold=0.5)
        
        if labels is None: #no prediction
            objectness_sum.append(1)
            weighted_variance_total.append(1)
            weighted_entropy_total.append(np.log(10))
            df.loc[len(df)] = [os.path.join(dir, folder, 'images', img), 1, 1, np.log(10)]

            continue
        
        cls_conf = np.array(cls_conf)
        # softmax cls_conf
        cls_conf = np.exp(cls_conf) / np.sum(np.exp(cls_conf), axis=1, keepdims=True)
        
        objectness_uncertainty = np.var(conf[conf > 0.5])
        # calculate entropy
        # objectness_entropy = np.apply_along_axis(lambda x: -1 * np.sum(x * np.log(x)), 1, cls_conf[conf > 0.5])
            
        
        entropy_cluster = []
        for i in range(cluster_num):
            cluster = cls_conf[conf > 0.5][labels == i]
            if len(cluster) == 0:
                entropy_cluster.append(0)
                continue
            entropy = np.apply_along_axis(lambda x: -1 * np.sum(x * np.log(x)), 1, cluster)
            entropy_cluster.append(np.mean(entropy))

        weighted_entropy = np.mean(np.array(entropy_cluster) * np.bincount(labels, minlength=cluster_num).reshape(-1, 1) / np.sum(np.bincount(labels)))
        
            # df = df.append({'dataset': dir, 'img_name': img, 'objectness_uncertainty': objectness_uncertainty, 'weighted_variance_sum': weighted_variance_sum, 'weighted_entropy': weighted_entropy}, ignore_index=True)
        df.loc[len(df)] = [os.path.join(dir, folder, 'images', img), objectness_uncertainty, weighted_variance_sum, weighted_entropy]
        # df.to_csv(save_name, mode='w', header=True, index=False)
        
        objectness_sum.append(objectness_uncertainty)
        weighted_variance_total.append(weighted_variance_sum)
        weighted_entropy_total.append(weighted_entropy)
        
        # print(f'objectness_uncertainty: {objectness_uncertainty}, weighted_variance_sum: {weighted_variance_sum}, weighted_entropy: {weighted_entropy}')

    print(f'{dir}: objectness_uncertainty: {np.mean(objectness_sum)}, weighted_variance_sum: {np.mean(weighted_variance_total)}, weighted_entropy: {np.mean(weighted_entropy_total)}')

    return objectness_sum, weighted_variance_total, weighted_entropy_total, df



def uncertainty_one(times, model):
    dir = ['GC10-DET_brightness_'+str(i) for i in [-50, -30, 30, 50]]
    dir.append('GC10-DET')
    
    dir = [os.path.join('/Data4/student_zhihan_data/data',i) for i in dir]

    for i in range(len(os.listdir(os.path.join(dir[-1], 'test', 'images')))):
        for d in dir:
            img = os.listdir(os.path.join(d, "test", "images"))[i]
            label = os.listdir(os.path.join(d, "test", "labels"))[i]
            

            img_path = os.path.join(d, "test", "images", img)
            label_path = os.path.join(d, "test", "labels", label)
        
            results = model([img_path for i in range(times)], verbose=False)
            # results = model([img_path for i in range(times)])
            
            boundingboxes = []
            boxes = []
            conf = []
            cls_conf = []
            for re in results:
                conf.extend(re.boxes.conf.cpu())
                tmp = re.boxes.xywhn
                boxes.extend(re.boxes.cpu())
                boundingboxes.extend(tmp.cpu())
                cls_conf.extend(re.cls_all.cpu())
                
            #read txt file
            cluster_num = len(np.loadtxt(label_path))
            boundingboxes = np.array(boundingboxes)
            conf = np.array(conf)
            cls_conf = np.array(cls_conf)
            selected_data, labels, variances, weighted_variance_sum = cluster_bounding_boxes(boundingboxes, n_clusters=cluster_num, confs=np.array(conf), threshold=0.5)
            
            objectness_uncertainty = np.var(conf[conf > 0.5])

            entropy_cluster = []
            for i in range(cluster_num):
                cluster = cls_conf[conf > 0.5][labels == i]
                entropy = np.apply_along_axis(lambda x: -1 * np.sum(x * np.log2(x)), 1, cluster)
                entropy_cluster.append(np.mean(entropy))

            weighted_entropy = np.mean(np.array(entropy_cluster) * np.bincount(labels, minlength=cluster_num).reshape(-1, 1) / np.sum(np.bincount(labels)))
            
            print(f'{d}: objectness_uncertainty: {objectness_uncertainty}, weighted_variance_sum: {weighted_variance_sum}, weighted_entropy: {weighted_entropy}')
        
        break

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='rightness,Gaussian,Sharpen_')
    parser.add_argument('--exclude', type=int, default=0)
    args = parser.parse_args()
    
    
    import pandas as pd
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    
    # model = YOLO('yolov8-dropblock.yaml').load('/Data4/student_zhihan_data/source_code/yolo/ultralytics/runs/detect/GC10-DET_brightness_0_detect_by_yolov8n_with_dropblock(p=0.05 s=5)2/weights/best.pt')
    model = YOLO('/Data4/student_zhihan_data/source_code/yolo/ultralytics/runs/detect/GC10-DET_brightness_0 detect by yolov8n with dropout(p=0.1)/weights/best.pt')
    # dir = '/Data4/student_zhihan_data/data/GC10-DET'
    # uncertainty_one(400, model)
    
    # dir = ['GC10-DET']
    # dir.extend(['GC10-DET_MedianBlur_'+str(i) for i in [15, 29, 43, 57, 71]])
    # dir.extend(['GC10-DET_BilateralBlur_'+str(i) for i in [60, 120, 180, 240, 300]])
    # dir.extend(['GC10-DET_Sharpening_0.5',
    #             'GC10-DET_Sharpening_1.5',
    #             'GC10-DET_Sharpening_2.0',
    #             'GC10-DET_Sharpening_2.5',
    #             'GC10-DET_Sharpening_3'])
    # dir.extend(['GC10-DET_Transform_Scale_'+i for i in ["0.0:0.05", "0.05:0.1", "0.1:0.15", "0.15:0.2", "0.2:0.25", "0.25:0.3"]])
    # dir.extend(['GC10-DET_brightness_'+str(i) for i in [-150, -100, -50, -30, -20, -15, -10, 10, 20, 30, 50, 60, 70, 90]])

    # dir = [f'GC10-DET_Sharpen_{i}' for i in [5]]
    # dir = [i for i in os.listdir('/Data4/student_zhihan_data/data') if 'Gaussian' in i and '.csv' not in i]
    #reverse dir
    # dir = dir[::-1]
    
    # model_paths = []
    # model_paths.extend(['/Data4/student_zhihan_data/source_code/yolo/ultralytics/runs/detect/GC10-DET_Sharpening_0.5_detect_by_yolov8n_with_dropblock(p=0.05/weights/best.pt',
    #                     '/Data4/student_zhihan_data/source_code/yolo/ultralytics/runs/detect/GC10-DET_Sharpening_1.5_detect_by_yolov8n_with_dropblock(p=0.05/weights/best.pt',
    #                     '/Data4/student_zhihan_data/source_code/yolo/ultralytics/runs/detect/GC10-DET_Sharpening_2.0_detect_by_yolov8n_with_dropblock(p=0.05/weights/best.pt',
    #                     '/Data4/student_zhihan_data/source_code/yolo/ultralytics/runs/detect/GC10-DET_Sharpening_2.5_detect_by_yolov8n_with_dropblock(p=0.05/weights/best.pt',
    #                     '/Data4/student_zhihan_data/source_code/yolo/ultralytics/runs/detect/GC10-DET_Sharpening_3_detect_by_yolov8n_with_dropblock(p=0.05/weights/best.pt'])
    
    # dir.extend(['/Data4/student_zhihan_data/data/GC10-DET_Transform_Scale_0.05:0.1', '/Data4/student_zhihan_data/data/GC10-DET_Transform_Scale_0.1:0.15000000000000002', '/Data4/student_zhihan_data/data/GC10-DET_Transform_Scale_0.15000000000000002:0.2', '/Data4/student_zhihan_data/data/GC10-DET_Transform_Scale_0.2:0.25', '/Data4/student_zhihan_data/data/GC10-DET_Transform_Scale_0.25:0.3'])
    # dir.append('GC10-DET')
    
    # dir.append('GC10-DET')
    selected_list = args.dataset.split(',')

    dir = [i for i in os.listdir('/Data4/student_zhihan_data/data') if '.csv' not in i and 'GC10-DET' in i and 'zip' not in i]
    for i in selected_list:
        selected_dir = [j for j in dir if i in j]
        
    if 'GC10-DET' in selected_dir and args.exclude == 0:
        selected_dir.remove('GC10-DET')
    
        
    dirs = [os.path.join('/Data4/student_zhihan_data/data',i) for i in selected_dir]
    

    dirs = ['/Data4/student_zhihan_data/data/GC10-DET']
    
    # df = pd.DataFrame(columns=['dataset', 'objectness_uncertainty', 'weighted_variance_sum', 'weighted_entropy'])
    # for idx, i in enumerate(selected_dir):
    #     print(i)
    #     df_img = pd.DataFrame(columns=['img_name', 'objectness_uncertainty', 'weighted_variance_sum', 'weighted_entropy'])
    #     save_name = f'{selected_dir[idx]}.csv'
    #     df_img.to_csv(save_name, mode='a', header=True, index=False)
    #     a, c, d, df_img = uncertainty(dirs[idx], 400, model, df_img, save_name)
    #     df_img.to_csv(save_name, mode='w', header=True, index=False)
    #     df.loc[len(df)] = [i, np.mean(a), np.mean(c), np.mean(d)]
    #     df.to_csv('Uncertainty.csv', mode='a', header=True, index=False)

    df = pd.DataFrame(columns=['dataset', 'objectness_uncertainty', 'weighted_variance_sum', 'weighted_entropy'])
    for idx, i in enumerate(dirs):
        df_img = pd.DataFrame(columns=['img_name', 'objectness_uncertainty', 'weighted_variance_sum', 'weighted_entropy'])
        save_name = 'GC10-DET_train.csv'
        df_img.to_csv(save_name, mode='a', header=True, index=False)
        a, c, d, df_img = uncertainty(dirs[idx], 400, model, df_img, save_name, folder='train')
        df_img.to_csv(save_name, mode='w', header=True, index=False)
        df.loc[len(df)] = [i, np.mean(a), np.mean(c), np.mean(d)]
        df.to_csv('Uncertainty.csv', mode='a', header=True, index=False)

