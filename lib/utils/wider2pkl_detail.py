import sys
import argparse
import cv2, os
import numpy as np
import scipy.io
import cPickle

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Convert WIDER_Face dataset annotation into pkl files')
    parser.add_argument('--img_root', dest='img_root',
                        help='path to wider face dataset images root directory',
                        default='./WIDER_train/images', type=str)
    parser.add_argument('--label', dest='label',
                        help='path to wider dataset annotation',
                        default='wider_face_train.mat', type=str)
    parser.add_argument('--out', dest='out_name',
                        help='output pkl file name',
                        default='output.pkl', type=str)
    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    try:
        wider_root = args.img_root
        label_file = args.label
    except:
        print 'Dataset root or annotation file does not exist!'
        sys.exit
        
    cache_file = args.out_name
    class_sets = ('face',)
    gt_roidb = []
    
    mat_read = scipy.io.loadmat(label_file)
    event_list = mat_read['event_list']
    file_name_list = mat_read['file_list']
    face_bbx_list = mat_read['face_bbx_list']
    blur_label_list = mat_read['blur_label_list']
    expression_label_list = mat_read['expression_label_list']
    illumination_label_list = mat_read['illumination_label_list']
    occlusion_label_list = mat_read['occlusion_label_list']
    pose_label_list = mat_read['pose_label_list']
    invalid_label_list = mat_read['invalid_label_list']

    for ind, face_bbx in enumerate(face_bbx_list):
        dirname = str(event_list[ind][0][0])
        for find, gt_entry in enumerate(face_bbx):
            for sind, sid in enumerate(gt_entry):
                file_name = str(file_name_list[ind][find][sind][0][0])
                img_file = os.path.join(wider_root, dirname, file_name+'.jpg')
                blur_box_list = np.ravel(blur_label_list[ind][find][sind][0])
                expression_box_list = np.ravel(expression_label_list[ind][find][sind][0])
                illumination_box_list = np.ravel(illumination_label_list[ind][find][sind][0])
                occlusion_box_list = np.ravel(occlusion_label_list[ind][find][sind][0])
                pose_box_list = np.ravel(pose_label_list[ind][find][sind][0])
                invalid_box_list = np.ravel(invalid_label_list[ind][find][sind][0])
                try:
                    img = cv2.imread(img_file)
                    img_size = img.shape
                    img_width = img_size[1]
                    img_height = img_size[0]
                    img_depth = img_size[2]
                except:
                    print 'Cant read image!'
                    sys.exit(1)
                for ssind, ssid in enumerate(sid):
                    boxes = np.zeros((len(ssid), 4), dtype=np.int32)
                    for sssind, sssid in enumerate(ssid):
                        x1 = sssid[0]
                        y1 = sssid[1]
                        w = sssid[2]
                        h = sssid[3]
                        x2 = x1 + w - 1
                        y2 = y1 + h - 1
                        x1 = max(x1 - 1, 0)
                        y1 = max(y1 - 1, 0)
                        boxes[sssind, :] = [x1, y1, x2, y2]

                info = {'image':img_file, 
                        'boxes':boxes, 
                        'width':img_width, 
                        'height':img_height, 
                        'depth':img_depth,
                        'blur': blur_box_list,
                        'expression': expression_box_list,
                        'illumination': illumination_box_list,
                        'occlusion': occlusion_box_list,
                        'pose': pose_box_list,
                        'invalid': invalid_box_list,
                        'gt_classes': 'face'}
                gt_roidb.append(info)

    with open(cache_file, 'wb') as fid:
        cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
    print 'Wrote gt roidb to {}'.format(cache_file)
    fid.close()
