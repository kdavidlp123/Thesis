# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import json

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Train a pose model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='whether to auto scale the learning rate according to the '
        'actual batch size and the original batch size.')
    parser.add_argument(
        '--show-dir',
        help='directory where the visualization images will be saved.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to display the prediction results in a window.')
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='visualize per interval samples.')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='display time of every window. (second)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.get('type', 'OptimWrapper')
        assert optim_wrapper in ['OptimWrapper', 'AmpOptimWrapper'], \
            '`--amp` is not supported custom optimizer wrapper type ' \
            f'`{optim_wrapper}.'
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    # resume training
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # enable auto scale learning rate
    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    # visualization-
    if args.show or (args.show_dir is not None):
        assert 'visualization' in cfg.default_hooks, \
            'PoseVisualizationHook is not set in the ' \
            '`default_hooks` field of config. Please set ' \
            '`visualization=dict(type="PoseVisualizationHook")`'

        cfg.default_hooks.visualization.enable = True
        cfg.default_hooks.visualization.show = args.show
        if args.show:
            cfg.default_hooks.visualization.wait_time = args.wait_time
        cfg.default_hooks.visualization.out_dir = args.show_dir
        cfg.default_hooks.visualization.interval = args.interval

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg





def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.isdir(dst):
        os.makedirs(dst)
    else:
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks, ignore)
            else:
                shutil.copy2(s, d)


def create_annotation_file(ear_type):
    if ear_type == "free":
        files = {}
        files["info"] = {}
        files["info"]["year"] = 2022
        files["info"]["version"] = "1"
        files["info"]["description"] = "for resnet backbone"
        files["licenses"] = []
        files["licenses"].append({"id" : 0, "name" : "no", "url" : "null"})
        files["images"] = []
        files["annotations"] = []
        files["categories"] = [{"supercategory" : "hand", 
                            "id" : 1, 
                            "name" : "ear", 
                            "keypoints" : ['w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 
                                            'w7', 'w8', 'w9', 'w10', 'w11', 'w12', 'w13', 
                                            'w14', 'w15', 'w16', 'w17', 'w18', 'w19', 'w20'],
                            "skeleton" : []}]
    else:
        files = {}
        files["info"] = {}
        files["info"]["year"] = 2022
        files["info"]["version"] = "1"
        files["info"]["description"] = "for resnet backbone"
        files["licenses"] = []
        files["licenses"].append({"id" : 0, "name" : "no", "url" : "null"})
        files["images"] = []
        files["annotations"] = []
        files["categories"] = [{"supercategory" : "hand", 
                           "id" : 1, 
                           "name" : "ear", 
                           "keypoints" : ['w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 
                                          'w7', 'w8', 'w9', 'w10', 'w11', 'w12', 'w13', 
                                          ],
                           "skeleton" : []}]
    return files

def create_dataset(names, kpt_img_path, kpt_json_path, root_eartype, degrees, ear_type, use_bbox = True, dataset_type = "training"):
    img_index = 0
    anno_file = create_annotation_file(ear_type)
    jsons = os.path.join(root_eartype, "2_json")
    return_to_ori_size = os.path.join(root_eartype, "4_return_to_ori_size")
    mmdet_bboxes_json = os.path.join(root_eartype, "5_mmdet_bboxes_json")

    
    if not os.path.isdir(kpt_img_path):
        os.makedirs(kpt_img_path)
    else:
        shutil.rmtree(kpt_img_path)
        os.makedirs(kpt_img_path)

    if not os.path.isdir(kpt_json_path):
        os.makedirs(kpt_json_path)
    else:
        shutil.rmtree(kpt_json_path)
        os.makedirs(kpt_json_path)

    for name in names:
        print(name)

        jsons_name = os.path.join(jsons, name)
        return_to_ori_size_name = os.path.join(return_to_ori_size, name)
        mmdet_bboxes_json_name = os.path.join(mmdet_bboxes_json, name)
        for deg in degrees:
            jsons_name_deg = os.path.join(jsons_name, deg)
            return_to_ori_size_name_deg = os.path.join(return_to_ori_size_name, deg)
            mmdet_bboxes_json_name_deg = os.path.join(mmdet_bboxes_json_name, deg)

            if os.path.exists(os.path.join(jsons_name_deg, "keypoint_location.json")):
            

                imgs = os.listdir(return_to_ori_size_name_deg)
                imgs = sorted(imgs, key = lambda s : int(os.path.splitext(os.path.basename(s))[0]), reverse = False)


                with open(os.path.join(mmdet_bboxes_json_name_deg,"bbox.json"), "r") as f_bbox:
                    bbox = json.load(f_bbox)
                with open(os.path.join(jsons_name_deg,"visible.json"), "r") as f_visible:
                    visible = json.load(f_visible)
                with open(os.path.join(jsons_name_deg,"keypoint_location.json"), "r") as f_keypoint_location:
                    keypoint_location = json.load(f_keypoint_location)



                current_jjson_frame_index = 0
                for img in imgs:
                    src = os.path.join(return_to_ori_size_name_deg, img)
                    dst =os.path.join(kpt_img_path, "{file_name}.png".format(file_name = dataset_type + "_" + str(img_index)))
                    shutil.copyfile(src, dst)
                    
                    images_info =  {"id" : img_index,
                                    "width": 540, 
                                    "height": 900, 
                                    "file_name": "{file_name}.png".format(file_name = dataset_type + "_" + str(img_index)), 
                                    "license": 0 }
                    keypoints_visible = []
                    bounding_box = []


                    if ear_type == "free":
                    
                        for l in range(0,21):
                            keypoints_visible.append(keypoint_location["frame_"+str(current_jjson_frame_index)][str(l)]["keypoint_location"][0])
                            keypoints_visible.append(keypoint_location["frame_"+str(current_jjson_frame_index)][str(l)]["keypoint_location"][1])
                            keypoints_visible.append(visible["frame_"+str(current_jjson_frame_index)][str(l)]["visible"])
                        if use_bbox:
                            bounding_box = bbox[str(current_jjson_frame_index)]
                        else:
                            bounding_box = [0, 0, 540, 960]
                        image_annotations = {"num_keypoints": 21 , 
                                            "keypoints": keypoints_visible, 
                                            "image_id":img_index, 
                                            "bbox":bounding_box, 
                                            "category_id": 1, 
                                            "id": img_index,  
                                            "area": bounding_box[2]*bounding_box[3],
                                            "iscrowd": 0}

                        anno_file["images"].append(images_info)
                        anno_file["annotations"].append(image_annotations)
                    else:

                        for l in range(0,14):
                            keypoints_visible.append(keypoint_location["frame_"+str(current_jjson_frame_index)][str(l)]["keypoint_location"][0])
                            keypoints_visible.append(keypoint_location["frame_"+str(current_jjson_frame_index)][str(l)]["keypoint_location"][1])
                            keypoints_visible.append(visible["frame_"+str(current_jjson_frame_index)][str(l)]["visible"])
                        if use_bbox:
                            bounding_box = bbox[str(current_jjson_frame_index)]
                        else:
                            bounding_box = [0, 0, 540, 960]
                        image_annotations = {"num_keypoints": 14 , 
                                            "keypoints": keypoints_visible, 
                                            "image_id":img_index, 
                                            "bbox":bounding_box, 
                                            "category_id": 1, 
                                            "id": img_index,  
                                            "area": bounding_box[2]*bounding_box[3],
                                            "iscrowd": 0}

                        anno_file["images"].append(images_info)
                        anno_file["annotations"].append(image_annotations)

                    current_jjson_frame_index += 1
                    img_index += 1 
    with open(os.path.join(kpt_json_path, dataset_type+".json"), "w") as f_anno_file:
        json.dump(anno_file, f_anno_file, indent = 4)        



def main():
    
    root = "../keypoint"
    save_root = "../MAT_inpainting"
    ear_types = ["free", "attached"]
    degrees = ['15cm_0mm_0deg', '15cm_25mm_5deg', '15cm_50mm_10deg', '20cm_0mm_0deg', '20cm_25mm_5deg', '20cm_50mm_10deg']
    for ear_type in ear_types:
        
        args = parse_args()
        if ear_type == "free":
            vars(args)['config'] = vars(args)['config']+"_free.py"
        else:
            vars(args)['config'] = vars(args)['config'] + "_attached.py"


        root_eartype = os.path.join(root, ear_type)
        names = os.listdir(os.path.join(root_eartype,"0_original_video"))
        
        for name in names:
            # save_model_path = os.path.join(save_root, ear_type, "model_save", name)
            # if not os.path.isdir(save_model_path):
            #     os.makedirs(save_model_path)
            # else:
            #     shutil.rmtree(save_model_path)
            #     os.makedirs(save_model_path)
            # vars(args)['work_dir'] = save_model_path
            
            names_copy = names.copy()
            names_copy.remove(name)


            # training_kpt_img_path = os.path.join(root_eartype, "5_training_img")
            # training_kpt_json_path = os.path.join(root_eartype, "5_training_json")
            # create_dataset(names_copy, training_kpt_img_path, training_kpt_json_path, root_eartype, degrees, ear_type,use_bbox = True, dataset_type = "training")


            test_kpt_img_path = os.path.join(root_eartype, "5_test_img")
            test_kpt_json_path = os.path.join(root_eartype, "5_test_json")
            create_dataset([name], test_kpt_img_path, test_kpt_json_path, root_eartype, degrees, ear_type, use_bbox = True, dataset_type = "test")



            # k_fold_result = os.path.join("k_fold", "result")
            # copytree(test_kpt_img_path, os.path.join(k_fold_result, ear_type, name,"5_test_img"))
            # copytree(test_kpt_json_path, os.path.join(k_fold_result, ear_type, name,"5_test_json"))
            
            copytree(test_kpt_img_path, os.path.join(save_root, ear_type, "result", name,"test_img"))
            copytree(test_kpt_json_path, os.path.join(save_root, ear_type, "result", name,"test_json"))


                    
                    # # load config
                    # cfg = Config.fromfile(args.config)

                    # # merge CLI arguments to config
                    # cfg = merge_args(cfg, args)

                    # # set preprocess configs to model
                    # if 'preprocess_cfg' in cfg:
                    #     cfg.model.setdefault('data_preprocessor',
                    #                         cfg.get('preprocess_cfg', {}))

                    # # build the runner from config
                    # runner = Runner.from_cfg(cfg)

                    # # start training
                    # runner.train()

                    # try:
                    #     shutil.rmtree(training_kpt_img_path)
                    #     shutil.rmtree(training_kpt_json_path)
                    #     shutil.rmtree(test_kpt_img_path)
                    #     shutil.rmtree(test_kpt_json_path)
                    #     print('Folder and its content removed') # Folder and its content removed
                    # except:
                    #     print('Folder not deleted')
            


        



if __name__ == '__main__':
    main()
