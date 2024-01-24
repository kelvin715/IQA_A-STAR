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
    
    if len(selected_data) < n_clusters:
        return selected_data, None, None, None

    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, max_iter=1000).fit(selected_data)
    labels = kmeans.labels_

    variances = []
    cls_entropy = []
    for i in range(n_clusters):
        cluster_data = selected_data[labels == i]
        variances.append(np.var(cluster_data, axis=0))
        
    
    weighted_variance = np.array(variances) * np.bincount(labels, minlength=n_clusters).reshape(-1, 1) / np.sum(np.bincount(labels))
    weighted_variance_sum = np.sum(np.mean(weighted_variance, axis=1))
  
    return selected_data, labels, variances, weighted_variance_sum

def uncertainty(dir, times, model):
    images = os.listdir(os.path.join(dir, "test", "images"))
    labels = os.listdir(os.path.join(dir, "test", "labels"))
    
    objectness_sum = []
    weighted_variance_total = []
    weighted_entropy_total = []
    for img, label in tqdm(zip(images, labels), total=len(images), desc="testing"):
        img_path = os.path.join(dir, "test", "images", img)
        label_path = os.path.join(dir, "test", "labels", label)
    
        results = model([img_path for i in range(times)], verbose=False)
        
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
        
        if labels is None:
            continue
        
        objectness_uncertainty = np.var(conf[conf > 0.5])
        
        entropy_cluster = []
        for i in range(cluster_num):
            cluster = cls_conf[conf > 0.5][labels == i]
            if len(cluster) == 0:
                entropy_cluster.append(0)
                continue
            entropy = np.apply_along_axis(lambda x: -1 * np.sum(x * np.log2(x)), 1, cluster)
            entropy_cluster.append(np.mean(entropy))

        weighted_entropy = np.mean(np.array(entropy_cluster) * np.bincount(labels, minlength=cluster_num).reshape(-1, 1) / np.sum(np.bincount(labels)))
        
        objectness_sum.append(objectness_uncertainty)
        weighted_variance_total.append(weighted_variance_sum)
        weighted_entropy_total.append(weighted_entropy)
        
        # print(f'objectness_uncertainty: {objectness_uncertainty}, weighted_variance_sum: {weighted_variance_sum}, weighted_entropy: {weighted_entropy}')

    print(f'{dir}: objectness_uncertainty: {np.mean(objectness_sum)}, weighted_variance_sum: {np.mean(weighted_variance_total)}, weighted_entropy: {np.mean(weighted_entropy_total)}')

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
    
    # model = YOLO('yolov8-dropblock.yaml').load('/Data4/student_zhihan_data/source_code/yolo/ultralytics/runs/detect/GC10-DET_brightness_0_detect_by_yolov8n_with_dropblock(p=0.05 s=5)2/weights/best.pt')
    model = YOLO('/Data4/student_zhihan_data/source_code/yolo/ultralytics/runs/detect/GC10-DET_brightness_0 detect by yolov8n with dropout(p=0.1)/weights/best.pt')
    # dir = '/Data4/student_zhihan_data/data/GC10-DET'
    # uncertainty_one(400, model)
    
    dir = ['GC10-DET_brightness_'+str(i) for i in [-15, 60, 70]]
    # dir.append('GC10-DET')
    
    dir = [os.path.join('/Data4/student_zhihan_data/data',i) for i in dir]
    for i in dir:
        uncertainty(i, 400, model)