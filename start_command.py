#     python code_training/main.py --mode train --batch_size 2 --validation_dataset_file_path  validation.tfrecords --experiment_name model --data_dir ../data/ctdar19_B2_m/train/ --NUM_STACKS 2 --num_epochs 10 --gpu 3

#     python code_training/main.py --mode test --batch_size 1 --experiment_name model --test_data_dir ../data/ctdar19_B2_m/test/SCAN/img/ --test_name test --test_scan True

#     python code_training/main.py --mode test --batch_size 1 --experiment_name model --test_data_dir ../data/ctdar19_B2_m/train_for_test/image --test_name test --test_scan True